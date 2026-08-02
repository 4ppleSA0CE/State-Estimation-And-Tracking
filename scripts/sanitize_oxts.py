#!/usr/bin/env python3
"""Make KITTI extract OXTS drives loadable: repair sample order, then keep the longest run.

KITTI's unsynced ("extract") OXTS streams emit occasional packets early, stamped ~70 ms in the
future and carrying a byte-identical copy of a packet that arrives a few samples later. The
high-rate loader refuses any non-strictly-increasing timestamp, so 7 of our 8 drives will not
load at all.

Ordering repair is minimal removal: keep a longest strictly increasing subsequence. A
running-maximum filter would instead keep the spurious future-stamped packet and discard the
good samples behind it, which is exactly backwards. Genuine OXTS outages are handled after
that, by splitting the repaired series at gaps over 200 ms and keeping the segment with the
most samples.

Ordering repair is cheap; segment selection is not. Across the 8 default drives ordering costs
0.6% of samples but segment selection costs another ~8%, almost all of it on 0009 and 0117,
which retain roughly 59% and 53%. The report states both, so the small number cannot stand in
for the large one.

The result is written as a derived drive `<date>_drive_<drive>_clean_extract`, which the
unmodified loader picks up from a drive id of "<drive>_clean" (its normalized_drive() is
zfill(4), a no-op on a string that long). Timestamp lines and packet files are byte copies of
the originals; only which samples appear, and how the packet files are numbered, changes.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROTOTYPES = Path(__file__).resolve().parents[1] / "prototypes" / "python"
if str(_PROTOTYPES) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPES))

# The loader truncates fractional seconds to microseconds, and that truncation is what turns
# some near-duplicate stamps into exact ties. Reusing its parser is the only way to repair the
# series the loader will actually see rather than a higher-precision one it never sees.
from kitti_highrate_loader import (  # noqa: E402
    HighRateOxtsConfig,
    _parse_timestamp,
    load_highrate_oxts,
)

# 200 ms is 20x the 10 ms nominal OXTS period, so "gap-free" here only means "no hole big
# enough to look like an outage". Sub-threshold holes survive into the kept segment and are
# real open-loop intervals the filter must coast through: 0117 keeps a 190 ms one. They are
# reported per drive as max_intra_segment_gap_s / n_intra_segment_gaps_over_50ms rather than
# cut, because tightening this would shorten the usable segments, and they will show up in RPE.
MAX_GAP_S = 0.2  # split the repaired series here; longer means a genuine OXTS outage
INTRA_SEGMENT_GAP_REPORT_S = 0.05  # 5x nominal; count surviving holes at least this large
LOW_RETENTION_PERCENT = 95.0  # name any drive retaining less than this in the printed summary
CLEAN_SUFFIX = "_clean"  # derived drive id suffix, e.g. 0015 -> 0015_clean
REPORT_PATH = Path("data/results/oxts_repair.csv")
DEFAULT_ROOT = Path("data/kitti_raw")

DEFAULT_DRIVES = (
    ("2011_09_26", "0001"),
    ("2011_09_26", "0009"),
    ("2011_09_26", "0015"),
    ("2011_09_26", "0117"),
    ("2011_09_30", "0020"),
    ("2011_09_30", "0033"),
    ("2011_10_03", "0042"),
    ("2011_09_29", "0004"),
)

REPORT_COLUMNS = (
    "date",
    "drive",
    "clean_drive",
    "n_original",
    "n_reordered_drops",
    "reorder_drop_percent",
    "n_gaps_over_200ms",
    "segment_start_index",
    "segment_end_index",
    "n_kept",
    "retention_percent",
    "duration_s",
    "duration_original_s",
    "max_intra_segment_gap_s",
    "n_intra_segment_gaps_over_50ms",
)


@dataclass(frozen=True)
class Repair:
    kept_indices: tuple[int, ...]  # indices into the ORIGINAL series, strictly increasing
    original_count: int  # samples in the source timestamps.txt
    dropped_count: int  # ORDERING removals only, original_count - len(longest subsequence)
    gap_count: int  # gaps over max_gap_s in the repaired series, before segment selection
    segment_start: int  # original index of the first sample of the chosen segment
    segment_end: int  # original index of the last sample of the chosen segment, inclusive
    duration_s: float  # span of the chosen segment
    duration_original_s: float  # span of the whole source series, for comparison
    max_intra_segment_gap_s: float  # largest step surviving INSIDE the chosen segment
    intra_segment_gaps_over_50ms: int  # steps inside it over INTRA_SEGMENT_GAP_REPORT_S

    @property
    def kept_count(self) -> int:
        return len(self.kept_indices)

    @property
    def reorder_drop_percent(self) -> float:
        """Share removed to repair ORDER. Segment selection discards more; see retention."""
        return 100.0 * self.dropped_count / self.original_count

    @property
    def retention_percent(self) -> float:
        """Share of the source that actually survives, ordering repair AND segment selection."""
        return 100.0 * self.kept_count / self.original_count


def increasing_subsequence(values) -> list[int]:
    """Indices of a longest STRICTLY increasing subsequence (patience sorting, O(n log n))."""
    tails: list[float] = []  # tails[k] = smallest possible tail of a subsequence of length k+1
    tail_index: list[int] = []
    predecessor = [-1] * len(values)
    for index, value in enumerate(values):
        slot = bisect.bisect_left(tails, value)  # bisect_left, not right: keeps it strict
        if slot == len(tails):
            tails.append(value)
            tail_index.append(index)
        else:
            tails[slot] = value
            tail_index[slot] = index
        predecessor[index] = tail_index[slot - 1] if slot > 0 else -1

    if not tail_index:
        return []
    chain: list[int] = []
    index = tail_index[-1]
    while index != -1:
        chain.append(index)
        index = predecessor[index]
    return chain[::-1]


def split_at_gaps(values, max_gap_s: float) -> list[tuple[int, int]]:
    """Half-open index ranges of `values` separated by steps larger than max_gap_s."""
    if not len(values):
        return []
    cuts = [0]
    cuts += [i + 1 for i in range(len(values) - 1) if values[i + 1] - values[i] > max_gap_s]
    cuts.append(len(values))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def repair(values, max_gap_s: float = MAX_GAP_S) -> Repair:
    """Repair sample order by minimal removal, then keep the longest gap-free segment."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot repair an empty timestamp series")

    kept = increasing_subsequence(values)
    kept_times = [values[index] for index in kept]
    segments = split_at_gaps(kept_times, max_gap_s)
    # Count first, elapsed time only as a tiebreak: OXTS is a fixed-rate sensor, so samples are
    # what an estimator consumes. A sparse segment spanning more wall-clock time is worse data,
    # not better, and must not beat a denser shorter one.
    start, stop = max(
        segments, key=lambda span: (span[1] - span[0], kept_times[span[1] - 1] - kept_times[span[0]])
    )
    chosen = kept[start:stop]
    steps = [b - a for a, b in zip(kept_times[start:stop], kept_times[start + 1:stop])]
    return Repair(
        kept_indices=tuple(chosen),
        original_count=len(values),
        dropped_count=len(values) - len(kept),
        gap_count=len(segments) - 1,
        segment_start=chosen[0],
        segment_end=chosen[-1],
        duration_s=values[chosen[-1]] - values[chosen[0]],
        duration_original_s=max(values) - min(values),
        max_intra_segment_gap_s=max(steps, default=0.0),
        intra_segment_gaps_over_50ms=sum(1 for step in steps if step > INTRA_SEGMENT_GAP_REPORT_S),
    )


def longest_increasing_length(values) -> int:
    """Length of a longest strictly increasing subsequence, from an O(n^2) DP.

    Kept deliberately independent of increasing_subsequence: a check that reused it would be
    tautological and blind to a bug in it.
    """
    array = np.asarray(values, dtype=float)
    count = array.shape[0]
    if count == 0:
        return 0
    best = np.ones(count, dtype=np.int64)
    for index in range(1, count):
        earlier = np.flatnonzero(array[:index] < array[index])
        if earlier.size:
            best[index] = int(best[earlier].max()) + 1
    return int(best.max())


def minimum_removals(values) -> int:
    """Same quantity as Repair.dropped_count, via the independent DP above."""
    return len(np.asarray(values, dtype=float)) - longest_increasing_length(values)


def clean_drive_id(drive: str) -> str:
    return f"{drive}{CLEAN_SUFFIX}"


def oxts_path(root: Path, date: str, drive: str) -> Path:
    return Path(root) / date / f"{date}_drive_{drive}_extract" / "oxts"


def read_timestamp_lines(oxts: Path) -> list[str]:
    text = oxts.joinpath("timestamps.txt").read_text(encoding="utf-8")
    if "\r" in text:
        raise ValueError(f"{oxts}/timestamps.txt has carriage returns; lines would not copy byte-exact")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{oxts}/timestamps.txt is empty")
    return lines


def timestamp_seconds(lines: list[str], oxts: Path) -> list[float]:
    parsed = [_parse_timestamp(line, oxts, number) for number, line in enumerate(lines, start=1)]
    return [(stamp - parsed[0]).total_seconds() for stamp in parsed]


def packet_paths(oxts: Path) -> list[Path]:
    return sorted(path for path in oxts.joinpath("data").glob("*.txt") if path.is_file())


def write_clean_drive(root: Path, date: str, drive: str, max_gap_s: float = MAX_GAP_S) -> Repair:
    """Repair one drive and write it as `<date>_drive_<drive>_clean_extract`."""
    source = oxts_path(root, date, drive)
    lines = read_timestamp_lines(source)
    packets = packet_paths(source)
    if len(lines) != len(packets):
        raise ValueError(
            f"{source}: {len(lines)} timestamps vs {len(packets)} packet files"
        )

    result = repair(timestamp_seconds(lines, source), max_gap_s)

    destination = oxts_path(root, date, clean_drive_id(drive))
    if destination.exists():
        shutil.rmtree(destination)  # a stale derived tree must never survive a rewrite
    destination.joinpath("data").mkdir(parents=True)
    destination.joinpath("timestamps.txt").write_text(
        "".join(f"{lines[index]}\n" for index in result.kept_indices), encoding="utf-8"
    )
    dataformat = source / "dataformat.txt"
    if dataformat.is_file():
        shutil.copyfile(dataformat, destination / "dataformat.txt")
    for position, index in enumerate(result.kept_indices):
        shutil.copyfile(packets[index], destination / "data" / f"{position:010d}.txt")
    return result


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_clean_drive(
    root: Path,
    date: str,
    drive: str,
    result: Repair,
    check_minimality: bool = True,
    max_gap_s: float = MAX_GAP_S,
) -> None:
    """Re-read both trees from disk and assert the derived drive is a faithful subsequence."""
    source = oxts_path(root, date, drive)
    destination = oxts_path(root, date, clean_drive_id(drive))
    source_lines = read_timestamp_lines(source)
    written_lines = read_timestamp_lines(destination)

    indices = list(result.kept_indices)
    assert indices == sorted(set(indices)), f"{drive}: kept indices are not strictly increasing"
    assert written_lines == [source_lines[i] for i in indices], (
        f"{drive}: written timestamps are not a byte-exact subsequence of the source"
    )
    seconds = timestamp_seconds(written_lines, destination)
    assert all(b > a for a, b in zip(seconds, seconds[1:])), (
        f"{drive}: written timestamps are not strictly increasing"
    )
    assert max((b - a for a, b in zip(seconds, seconds[1:])), default=0.0) <= max_gap_s, (
        f"{drive}: written segment still contains a gap over {max_gap_s} s"
    )

    source_packets = packet_paths(source)
    written_packets = packet_paths(destination)
    assert [path.name for path in written_packets] == [
        f"{i:010d}.txt" for i in range(len(indices))
    ], f"{drive}: packet files are not numbered contiguously from zero"
    for position, index in enumerate(indices):
        assert _digest(written_packets[position]) == _digest(source_packets[index]), (
            f"{drive}: packet {position} is not a byte copy of source packet {index}"
        )

    if check_minimality:
        source_seconds = timestamp_seconds(source_lines, source)
        expected = minimum_removals(source_seconds)
        assert result.dropped_count == expected, (
            f"{drive}: dropped {result.dropped_count} samples, theoretical minimum is {expected}"
        )
        # dropped_count is a PRE-segment-selection number: it says nothing about what
        # split_at_gaps and the tiebreak then threw away, so a bug in either would pass the
        # assertion above in silence. Check the series actually WRITTEN is maximal inside its
        # own index window - nothing in [segment_start, segment_end] could have been kept too.
        window = source_seconds[result.segment_start:result.segment_end + 1]
        best = longest_increasing_length(window)
        assert best == len(indices), (
            f"{drive}: wrote {len(indices)} samples from source window "
            f"[{result.segment_start}, {result.segment_end}], but {best} are maximal there; "
            f"the chosen segment is not maximal in its own window"
        )


def _format_table(rows: list[dict]) -> str:
    widths = {name: max(len(name), *(len(str(row[name])) for row in rows)) for name in REPORT_COLUMNS}
    lines = ["  ".join(name.rjust(widths[name]) for name in REPORT_COLUMNS)]
    for row in rows:
        lines.append("  ".join(str(row[name]).rjust(widths[name]) for name in REPORT_COLUMNS))
    return "\n".join(lines)


def _row(date: str, drive: str, result: Repair) -> dict:
    return {
        "date": date,
        "drive": drive,
        "clean_drive": clean_drive_id(drive),
        "n_original": result.original_count,
        "n_reordered_drops": result.dropped_count,
        "reorder_drop_percent": f"{result.reorder_drop_percent:.3f}",
        "n_gaps_over_200ms": result.gap_count,
        "segment_start_index": result.segment_start,
        "segment_end_index": result.segment_end,
        "n_kept": result.kept_count,
        "retention_percent": f"{result.retention_percent:.3f}",
        "duration_s": f"{result.duration_s:.3f}",
        "duration_original_s": f"{result.duration_original_s:.3f}",
        # microseconds, matching the loader's timestamp truncation: at 3 decimals 0004's
        # 50.037 ms hole would print as "0.050" and read as exactly on the 50 ms threshold.
        "max_intra_segment_gap_s": f"{result.max_intra_segment_gap_s:.6f}",
        "n_intra_segment_gaps_over_50ms": result.intra_segment_gaps_over_50ms,
    }


def _parse_drives(spec: str | None) -> tuple[tuple[str, str], ...]:
    if spec is None:
        return DEFAULT_DRIVES
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--drives entry must be date:drive, got {item!r}")
        date, drive = item.split(":", maxsplit=1)
        pairs.append((date.strip(), drive.strip()))
    if not pairs:
        raise ValueError("--drives selected no drives")
    return tuple(pairs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=HighRateOxtsConfig().cache_root)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--max-gap-s", type=float, default=MAX_GAP_S)
    parser.add_argument("--drives", default=None, help="comma-separated date:drive pairs")
    parser.add_argument("--skip-minimality-check", action="store_true",
                        help="skip the O(n^2) independent checks that removal is minimal and "
                             "that the written segment is maximal in its own index window")
    parser.add_argument("--skip-load-check", action="store_true",
                        help="skip loading each derived drive through the high-rate loader")
    args = parser.parse_args(argv)

    rows = []
    for date, drive in _parse_drives(args.drives):
        result = write_clean_drive(args.root, date, drive, args.max_gap_s)
        verify_clean_drive(args.root, date, drive, result,
                           not args.skip_minimality_check, args.max_gap_s)
        rows.append(_row(date, drive, result))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(_format_table(rows))
    total_original = sum(row["n_original"] for row in rows)
    total_reordered = sum(row["n_reordered_drops"] for row in rows)
    total_kept = sum(row["n_kept"] for row in rows)
    discarded = total_original - total_kept
    # Two different numbers, and the first one alone badly understates the loss: ordering
    # repair removes a fraction of a percent, segment selection removes an order more.
    print(f"\nordering removals: {total_reordered} of {total_original} samples "
          f"({100.0 * total_reordered / total_original:.3f}%)")
    print(f"overall retention: {total_kept} of {total_original} samples kept "
          f"({100.0 * total_kept / total_original:.3f}%), {discarded} discarded "
          f"({100.0 * discarded / total_original:.3f}%) once segment selection is counted")
    thin = [row for row in rows if float(row["retention_percent"]) < LOW_RETENTION_PERCENT]
    if thin:
        print(f"drives retaining under {LOW_RETENTION_PERCENT:.0f}%:")
        for row in thin:
            print(f"  {row['date']} {row['clean_drive']}: {row['n_kept']} of "
                  f"{row['n_original']} samples ({row['retention_percent']}%)")
    print(f"report: {args.report}")

    if not args.skip_load_check:
        print("\nhigh-rate loader check:")
        for row in rows:
            # force_refresh both bypasses any cache keyed on this derived drive - _validate_cache
            # checks metadata only, so one written under a different --max-gap-s would load
            # silently and this check would validate it instead of the tree just written - and
            # leaves a correct cache behind, since the loader rewrites it from the raw files.
            sequence = load_highrate_oxts(
                HighRateOxtsConfig(root=args.root, date=row["date"], drive=row["clean_drive"],
                                   cache_root=args.cache_root, force_refresh=True)
            )
            assert sequence.sample_count == row["n_kept"], (
                f"{row['clean_drive']}: loader saw {sequence.sample_count} samples, "
                f"wrote {row['n_kept']}"
            )
            print(f"  {row['date']} {row['clean_drive']}: "
                  f"{sequence.sample_count} samples, {sequence.duration_s:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
