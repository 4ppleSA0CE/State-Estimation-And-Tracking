// Arrival-order regression suite for PendingFrameQueue.
//
// These six sequences (S1..S6) are the orders a live DDS harness actually produced against
// detection_transform_node. Replaying them in-process is the point of the extraction: with the
// logic inline, four seeded defects -- including reintroducing the exact non-monotonic
// overtake the node exists to prevent -- left the whole workspace suite green.
//
// Two assertions carry most of the weight, and both are needed:
//   * the EXACT emitted sequence, because a mutant can stay monotonic and still be wrong
//     (pop_back-on-overflow emits [0, 2] instead of [1, 2] -- monotonic, and wrong);
//   * strict stamp monotonicity plus per-emit pose identity, checked by the rig on EVERY emit
//     of EVERY scenario, because an overtake is a whole-run property, not a per-step one.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "kf_tracker/pending_frames.hpp"

namespace {

using kf_tracker::DropReason;
using kf_tracker::DroppedFrame;
using kf_tracker::EgoPose;
using kf_tracker::EgoPoseBuffer;
using kf_tracker::PendingFrameQueue;

using V = std::vector<int>;

// Frame k is stamped 1000 s + 10 ms * k -- the pipeline's real 100 Hz IMU grid, expressed in
// the int64 nanoseconds every pipeline stamp comparison uses. k may be negative: S5 below needs
// a frame that predates everything the ego buffer will ever hold.
constexpr std::int64_t kBaseNs = 1000LL * 1000000000LL;
constexpr std::int64_t kStepNs = 10LL * 1000000LL;

std::int64_t stampOf(int k) { return kBaseNs + static_cast<std::int64_t>(k) * kStepNs; }

// The pose for frame k has x == k, so every emit can assert it received ITS OWN pose rather
// than merely some pose. A lookup mutant that still emits in the right order dies here.
EgoPose poseFor(int k) {
  EgoPose e;
  e.p = Eigen::Vector3d(static_cast<double>(k), 0.0, 0.0);
  return e;
}

// Replays one arrival order. det(k) / ego(k) return what that SINGLE arrival made publishable;
// emitted() accumulates the whole run for the exact-sequence assertions. Every emit is checked
// for stamp/pose identity and strict monotonicity as it passes through.
class Rig {
 public:
  Rig(std::size_t max_pending, std::size_t ego_capacity)
      : buffer_(ego_capacity), queue_(max_pending) {
    queue_.setDropObserver([this](const DroppedFrame& d) { drops_.push_back(d); });
  }

  V det(int k) {
    queue_.onFrame(stampOf(k), k, buffer_, out_);
    return harvest();
  }

  V ego(int k) {
    buffer_.insert(stampOf(k), poseFor(k));
    queue_.onEgo(buffer_, out_);
    return harvest();
  }

  const V& emitted() const { return emitted_; }
  const std::vector<DroppedFrame>& drops() const { return drops_; }
  std::uint64_t dropped() const { return queue_.dropped(); }
  std::size_t pending() const { return queue_.size(); }

 private:
  V harvest() {
    V ids;
    for (const auto& e : out_) {
      EXPECT_EQ(e.stamp, stampOf(e.frame)) << "emit " << e.frame << " carries a foreign stamp";
      EXPECT_DOUBLE_EQ(e.ego.p.x(), static_cast<double>(e.frame))
          << "emit " << e.frame << " was paired with another frame's ego pose";
      EXPECT_GT(e.stamp, last_stamp_) << "output went non-monotonic at frame " << e.frame;
      last_stamp_ = e.stamp;
      ids.push_back(e.frame);
      emitted_.push_back(e.frame);
    }
    return ids;
  }

  EgoPoseBuffer buffer_;
  PendingFrameQueue<int> queue_;
  std::vector<PendingFrameQueue<int>::Emit> out_;
  V emitted_;
  std::vector<DroppedFrame> drops_;
  std::int64_t last_stamp_ = std::numeric_limits<std::int64_t>::min();
};

// ---------------------------------------------------------------------------
// S1 -- the steady state, plus the one case that legitimately publishes on arrival.
// ---------------------------------------------------------------------------
TEST(PendingFrameQueue, S1AlternatingArrivalsAndTheGenuineFastPath) {
  Rig r(8, 8);
  for (int k = 0; k < 3; ++k) {
    EXPECT_EQ(r.det(k), V{}) << "det " << k << " must wait for its own ego state";
    EXPECT_EQ(r.ego(k), V{k}) << "ego " << k << " must release exactly its own frame";
  }
  // Ego BEFORE its detection: the queue is empty and the exact stamp is present, so this frame
  // -- and only this shape of frame -- publishes on arrival.
  EXPECT_EQ(r.ego(3), V{});
  EXPECT_EQ(r.det(3), (V{3}));
  EXPECT_EQ(r.emitted(), (V{0, 1, 2, 3}));
  EXPECT_EQ(r.dropped(), 0u);
  EXPECT_EQ(r.pending(), 0u);
}

// ---------------------------------------------------------------------------
// S2 -- THE OVERTAKE TRAP. The most important test in this file.
//
// det1's ego state IS already buffered when det1 arrives, and it must STILL queue, because
// det0 is ahead of it. Without the pending-empty guard this run emits [1, 0]: /detections_map
// goes non-monotonic and the tracker is stepped backwards in time.
// ---------------------------------------------------------------------------
TEST(PendingFrameQueue, S2ALaterFrameWhoseEgoIsPresentMustNotOvertakeAWaitingOne) {
  Rig r(8, 8);
  EXPECT_EQ(r.det(0), V{});
  EXPECT_EQ(r.ego(1), V{}) << "ego 1 cannot release frame 1 -- frame 0 is still ahead of it";
  EXPECT_EQ(r.det(1), V{}) << "frame 1's ego IS buffered here; it must queue behind frame 0";
  EXPECT_EQ(r.ego(0), (V{0, 1}));
  EXPECT_EQ(r.emitted(), (V{0, 1}));
  EXPECT_EQ(r.dropped(), 0u);
}

// ---------------------------------------------------------------------------
// S3 -- the head's ego state is merely LATE, not lost. Nothing may be published, and nothing
// may be dropped, while the buffer still has room: a not-yet-full buffer has not received that
// stamp YET. Dropping on stamp alone emits [1, 2, 3] and silently destroys frame 0.
// ---------------------------------------------------------------------------
TEST(PendingFrameQueue, S3ALateHeadBlocksEveryoneAndIsNotDropped) {
  Rig r(8, 8);
  for (int k = 0; k <= 3; ++k) EXPECT_EQ(r.det(k), V{}) << "det " << k;
  for (int k = 1; k <= 3; ++k)
    EXPECT_EQ(r.ego(k), V{}) << "ego " << k << " released a frame past the blocked head";
  EXPECT_EQ(r.dropped(), 0u) << "frame 0 was dropped while the ego buffer still had room";

  EXPECT_EQ(r.ego(0), (V{0, 1, 2, 3})) << "the late head must unblock the whole queue at once";
  EXPECT_EQ(r.emitted(), (V{0, 1, 2, 3}));
  EXPECT_EQ(r.dropped(), 0u);
  EXPECT_EQ(r.pending(), 0u);
}

// ---------------------------------------------------------------------------
// S4 -- ego states arriving in reverse. FIFO release order is a property of the QUEUE, not of
// the ego arrival order.
// ---------------------------------------------------------------------------
TEST(PendingFrameQueue, S4ReverseEgoBurstStillReleasesInStampOrder) {
  Rig r(8, 8);
  for (int k = 0; k <= 4; ++k) EXPECT_EQ(r.det(k), V{}) << "det " << k;
  for (int k = 4; k >= 1; --k) EXPECT_EQ(r.ego(k), V{}) << "ego " << k;
  EXPECT_EQ(r.ego(0), (V{0, 1, 2, 3, 4}));
  EXPECT_EQ(r.emitted(), (V{0, 1, 2, 3, 4}));
  EXPECT_EQ(r.dropped(), 0u);
}

// ---------------------------------------------------------------------------
// S4b -- the shape the real pipeline produces: eskf_node emits /ego/state for t_k only when
// IMU t_{k+1} arrives, so the ego stream trails detections by exactly one step. Steady state
// is therefore "one frame in the queue, released by the next ego state", forever.
// ---------------------------------------------------------------------------
TEST(PendingFrameQueue, S4bEgoTrailingByOneStepRunsAtAOneFrameLag) {
  Rig r(8, 16);
  for (int k = 0; k <= 7; ++k) {
    EXPECT_EQ(r.det(k), V{}) << "det " << k << " published without its own ego state";
    const V released = r.ego(k - 1);
    if (k == 0)
      EXPECT_EQ(released, V{}) << "nothing exists to release before frame 0's ego arrives";
    else
      EXPECT_EQ(released, V{k - 1}) << "ego " << (k - 1) << " must release exactly one frame";
    EXPECT_EQ(r.pending(), 1u) << "steady state is exactly one frame in flight";
  }
  EXPECT_EQ(r.ego(7), (V{7}));
  EXPECT_EQ(r.emitted(), (V{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(r.dropped(), 0u);
}

// ---------------------------------------------------------------------------
// S5 -- an unresolvable frame. Frame -100 predates the drive; its ego state will never come.
// It may only be dropped once the buffer is FULL and has provably evicted past its stamp --
// which happens on the third ego insert, not before. Until then it legitimately blocks.
// ---------------------------------------------------------------------------
TEST(PendingFrameQueue, S5UnresolvableHeadIsDroppedOnlyOnceTheBufferIsFull) {
  Rig r(8, 3);                    // ego capacity 3: full on the third insert
  EXPECT_EQ(r.det(-100), V{});
  EXPECT_EQ(r.det(1), V{});
  EXPECT_EQ(r.det(2), V{});
  EXPECT_EQ(r.ego(1), V{}) << "buffer not full yet -- frame -100 may still be waiting";
  EXPECT_EQ(r.ego(2), V{}) << "buffer not full yet -- frame -100 may still be waiting";
  EXPECT_EQ(r.dropped(), 0u) << "dropped an unmatched head before the buffer could fill";

  EXPECT_EQ(r.ego(3), (V{1, 2})) << "the insert that FILLS the buffer resolves the blockage";
  EXPECT_EQ(r.emitted(), (V{1, 2}));
  EXPECT_EQ(r.dropped(), 1u);
  ASSERT_EQ(r.drops().size(), 1u);
  EXPECT_EQ(r.drops()[0].stamp, stampOf(-100));
  EXPECT_EQ(r.drops()[0].reason, DropReason::kEgoUnreachable);
  EXPECT_EQ(r.drops()[0].ego_oldest_stamp, stampOf(1));
  EXPECT_EQ(r.pending(), 0u);
}

// ---------------------------------------------------------------------------
// S6 -- pending overflow evicts the OLDEST frame. Note what this asserts and why: evicting the
// NEWEST instead yields [0, 2], which is still monotonic and still "loses one frame", so a
// monotonicity-only assertion passes on it. Only the exact set separates the two.
// ---------------------------------------------------------------------------
TEST(PendingFrameQueue, S6OverflowEvictsTheOldestPendingFrameNotTheNewest) {
  Rig r(2, 8);
  EXPECT_EQ(r.det(0), V{});
  EXPECT_EQ(r.det(1), V{});
  EXPECT_EQ(r.det(2), V{}) << "the third frame overflows a 2-deep queue";
  EXPECT_EQ(r.dropped(), 1u);
  EXPECT_EQ(r.pending(), 2u);

  EXPECT_EQ(r.ego(0), V{}) << "frame 0 was evicted -- its ego state releases nothing";
  EXPECT_EQ(r.ego(1), (V{1}));
  EXPECT_EQ(r.ego(2), (V{2}));
  EXPECT_EQ(r.emitted(), (V{1, 2})) << "evicting the newest would give [0, 2] -- also monotonic";
  ASSERT_EQ(r.drops().size(), 1u);
  EXPECT_EQ(r.drops()[0].stamp, stampOf(0));
  EXPECT_EQ(r.drops()[0].reason, DropReason::kQueueOverflow);
}

// ---------------------------------------------------------------------------
// Supporting invariants
// ---------------------------------------------------------------------------

// The drain after the overflow eviction. The new head's pose may ALREADY be buffered, in which
// case it publishes now; without that drain it stalls until the next ego state, which under a
// 1-deep queue means it is evicted by the next frame and never published at all.
TEST(PendingFrameQueue, OverflowDrainsTheNewHeadImmediately) {
  Rig r(1, 8);
  EXPECT_EQ(r.det(0), V{});
  EXPECT_EQ(r.ego(1), V{}) << "frame 0 is ahead of frame 1 and has no pose yet";
  EXPECT_EQ(r.det(1), (V{1})) << "evicting frame 0 exposes a head whose pose is already held";
  EXPECT_EQ(r.emitted(), (V{1}));
  EXPECT_EQ(r.dropped(), 1u);
  EXPECT_EQ(r.pending(), 0u);
}

// The fast path is a shortcut, not a second code path with its own rules: an arriving frame
// whose stamp is absent queues even when the buffer holds neighbouring stamps.
TEST(PendingFrameQueue, FastPathRequiresTheExactStamp) {
  Rig r(8, 8);
  EXPECT_EQ(r.ego(0), V{});
  EXPECT_EQ(r.ego(2), V{});
  EXPECT_EQ(r.det(1), V{}) << "stamps 0 and 2 are held; stamp 1 is not, so frame 1 waits";
  EXPECT_EQ(r.pending(), 1u);
  EXPECT_EQ(r.ego(1), (V{1}));
  EXPECT_EQ(r.emitted(), (V{1}));
}

TEST(PendingFrameQueue, ZeroMaxPendingIsClampedToOne) {
  PendingFrameQueue<int> q(0);
  EXPECT_EQ(q.maxPending(), 1u);
  EgoPoseBuffer buffer(4);
  std::vector<PendingFrameQueue<int>::Emit> out;
  q.onFrame(stampOf(0), 0, buffer, out);
  EXPECT_EQ(q.size(), 1u) << "a clamp to 0 would discard every frame on arrival";
  EXPECT_EQ(q.dropped(), 0u);
}

// Drops are counted whether or not a diagnostic sink is installed -- the node's shutdown
// summary reads the counter, and it must not depend on a logger being attached.
TEST(PendingFrameQueue, CountsDropsWithNoObserverInstalled) {
  PendingFrameQueue<int> q(1);
  EgoPoseBuffer buffer(4);
  std::vector<PendingFrameQueue<int>::Emit> out;
  q.onFrame(stampOf(0), 0, buffer, out);
  q.onFrame(stampOf(1), 1, buffer, out);
  EXPECT_EQ(q.dropped(), 1u);
  EXPECT_EQ(q.size(), 1u);
}

// `out` is an owned output, not an accumulator: a caller reusing one vector across arrivals
// must never republish the previous arrival's frames.
TEST(PendingFrameQueue, OutIsClearedOnEveryCall) {
  EgoPoseBuffer buffer(8);
  PendingFrameQueue<int> q(8);
  std::vector<PendingFrameQueue<int>::Emit> out;

  q.onFrame(stampOf(0), 0, buffer, out);
  EXPECT_TRUE(out.empty());
  buffer.insert(stampOf(0), poseFor(0));
  q.onEgo(buffer, out);
  ASSERT_EQ(out.size(), 1u);
  q.onEgo(buffer, out);
  EXPECT_TRUE(out.empty()) << "the second drain re-reported the first drain's frame";
}

}  // namespace
