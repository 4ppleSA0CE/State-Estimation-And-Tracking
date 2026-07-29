// Arrival-order kernel for the detection/ego coupling: the bounded queue that holds each
// detection frame until the ego pose at its EXACT stamp is available, and releases frames in
// stamp order once it is.
//
// Extracted from detection_transform_node.cpp (design doc section 5.3) for one reason: this is
// the node's only nontrivial behaviour and it is a function of ARRIVAL ORDER, which a ROS test
// can only produce by racing two DDS topics. Four seeded defects in the inline version -- the
// pending-empty guard, the buffer-full conjunct, pop_back-on-overflow, and a missing drain --
// all left the workspace suite green. Here every order is replayable in-process.
//
// The two rules that are easy to get wrong, both load-bearing:
//
//   1. A frame publishes on arrival ONLY when the queue is empty. If anything is already
//      waiting, a frame whose ego pose has already arrived still queues behind it -- otherwise
//      it OVERTAKES the older frame and /detections_map goes non-monotonic, stepping the
//      tracker backwards in time. (det0, ego1, det1, ego0 emits [1, 0] without the guard.)
//   2. A frame is only unmatchable once the ego buffer is FULL and the frame predates its
//      oldest entry. A not-yet-full buffer simply has not received that stamp YET; dropping on
//      stamp alone destroys the head of the queue the instant any later ego pose shows up.
//
// ROS-independent: standard library + EgoPoseBuffer only, so the gtest host-compiles.
#ifndef KF_TRACKER_PENDING_FRAMES_HPP
#define KF_TRACKER_PENDING_FRAMES_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "kf_tracker/ego_transform.hpp"

namespace kf_tracker {

enum class DropReason {
  kQueueOverflow,    // the queue hit max_pending; the OLDEST frame is discarded
  kEgoUnreachable,   // the ego buffer is full and has already evicted past this stamp
};

struct DroppedFrame {
  std::int64_t stamp = 0;
  DropReason reason = DropReason::kQueueOverflow;
  std::int64_t ego_oldest_stamp = 0;   // buffer state at the drop, for the diagnostic
};

// Frame is whatever the caller wants paired with a pose (the ROS node passes a message
// ConstSharedPtr; the tests pass an int). It must be default-constructible and movable.
template <class Frame>
class PendingFrameQueue {
 public:
  struct Emit {
    std::int64_t stamp = 0;
    Frame frame{};
    EgoPose ego;      // the pose at EXACTLY `stamp`, copied -- the buffer may evict it later
  };

  // max_pending == 0 would mean "drop every frame on arrival", which is never what a caller
  // means; clamp to 1 as EgoPoseBuffer clamps its capacity. The node rejects it upstream.
  explicit PendingFrameQueue(std::size_t max_pending)
      : max_pending_(max_pending == 0 ? 1 : max_pending) {}

  // Optional diagnostic sink; drops are counted whether or not one is installed.
  void setDropObserver(std::function<void(const DroppedFrame&)> observer) {
    on_drop_ = std::move(observer);
  }

  // A detection frame arrived. `out` is CLEARED, then filled with everything that became
  // publishable, oldest first.
  void onFrame(std::int64_t stamp, Frame frame, const EgoPoseBuffer& ego,
               std::vector<Emit>& out) {
    out.clear();
    // Fast path ONLY when nothing is already waiting -- see rule 1 in the header comment.
    //
    // The GUARD is load-bearing; the fast path itself is not. Deleting this whole block
    // outright is an equivalent mutant (audit 2026-07-28, defect M6, survived all 12 tests by
    // construction): with an empty queue the overflow test cannot fire, so push_back followed
    // by drain() emits exactly this, one deque round-trip later. It is kept because the design
    // doc specifies it and it skips that round-trip on the common path -- not because any
    // behaviour depends on it. Do not "harden" it with a test; there is nothing to observe.
    if (pending_.empty()) {
      if (const EgoPose* pose = ego.find(stamp)) {
        out.push_back(Emit{stamp, std::move(frame), *pose});
        return;
      }
    }
    if (pending_.size() >= max_pending_) {
      ++dropped_;
      notify(DroppedFrame{pending_.front().stamp, DropReason::kQueueOverflow, ego.oldestStamp()});
      pending_.pop_front();
    }
    pending_.push_back(Entry{stamp, std::move(frame)});
    // The head may have just changed (overflow drop above) and its pose may already be
    // buffered -- one map lookup is cheaper than stalling that frame until the next EgoState.
    drain(ego, out);
  }

  // An ego pose arrived; the caller has ALREADY inserted it into `ego`. `out` is cleared.
  void onEgo(const EgoPoseBuffer& ego, std::vector<Emit>& out) {
    out.clear();
    drain(ego, out);
  }

  std::uint64_t dropped() const { return dropped_; }
  std::size_t size() const { return pending_.size(); }
  bool empty() const { return pending_.empty(); }
  std::size_t maxPending() const { return max_pending_; }

 private:
  struct Entry {
    std::int64_t stamp = 0;
    Frame frame{};
  };

  // Release every pending frame whose pose has now arrived, in FIFO (== stamp) order, and drop
  // the head when its pose can provably never arrive. Stop at the first frame still
  // legitimately waiting: everything behind it is newer and waits too.
  void drain(const EgoPoseBuffer& ego, std::vector<Emit>& out) {
    while (!pending_.empty()) {
      Entry& front = pending_.front();
      if (const EgoPose* pose = ego.find(front.stamp)) {
        out.push_back(Emit{front.stamp, std::move(front.frame), *pose});
        pending_.pop_front();
        continue;
      }
      if (ego.full() && front.stamp < ego.oldestStamp()) {
        ++dropped_;
        notify(DroppedFrame{front.stamp, DropReason::kEgoUnreachable, ego.oldestStamp()});
        pending_.pop_front();
        continue;
      }
      break;
    }
  }

  void notify(const DroppedFrame& drop) {
    if (on_drop_) on_drop_(drop);
  }

  std::size_t max_pending_;
  std::deque<Entry> pending_;
  std::uint64_t dropped_ = 0;
  std::function<void(const DroppedFrame&)> on_drop_;
};

}  // namespace kf_tracker

#endif  // KF_TRACKER_PENDING_FRAMES_HPP
