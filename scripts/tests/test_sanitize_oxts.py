"""Tests for the OXTS order/gap sanitizer. Synthetic series only: no network, no KITTI data.

The write/load tests build a tiny fake drive tree under tmp_path, so they exercise the real
frozen-loader contract without needing anything downloaded.
"""
import csv
import hashlib
import pathlib
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "prototypes" / "python"))

import sanitize_oxts  # noqa: E402
from sanitize_oxts import (  # noqa: E402
    INTRA_SEGMENT_GAP_REPORT_S,
    MAX_GAP_S,
    REPORT_COLUMNS,
    increasing_subsequence,
    longest_increasing_length,
    main,
    minimum_removals,
    repair,
    split_at_gaps,
    verify_clean_drive,
    write_clean_drive,
)


def _reference_lis_length(values):
    """Independent O(n^2) DP for the longest strictly increasing subsequence length.

    Deliberately not the algorithm under test: sharing a helper would make the minimality
    assertion blind to a bug in that helper.
    """
    if not values:
        return 0
    best = [1] * len(values)
    for i in range(len(values)):
        for j in range(i):
            if values[j] < values[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best)


def _assert_subsequence(kept, values):
    kept = list(kept)
    assert kept == sorted(set(kept)), "kept indices must be strictly increasing and unique"
    assert all(0 <= i < len(values) for i in kept), "kept indices must index the original series"
    picked = [values[i] for i in kept]
    assert all(b > a for a, b in zip(picked, picked[1:])), "kept series must be strictly increasing"


# --- ordering repair -------------------------------------------------------------------


def test_clean_series_is_returned_unchanged():
    values = [round(0.01 * i, 6) for i in range(200)]
    result = repair(values)
    assert result.dropped_count == 0
    assert list(result.kept_indices) == list(range(200))
    assert result.segment_start == 0
    assert result.segment_end == 199
    assert result.gap_count == 0
    assert result.duration_s == pytest.approx(1.99)


def test_future_spike_is_dropped_and_the_samples_it_shadows_survive():
    # Index 100 carries a timestamp 70 ms in the future that duplicates index 107. A
    # prefix-max/running-maximum filter keeps the spike and eats indices 101..106; minimal
    # removal drops the spike alone. Indices 101..106 must all survive.
    values = [round(0.01 * i, 6) for i in range(200)]
    values[100] = values[107]
    result = repair(values)

    assert result.dropped_count == 1, "only the spurious packet should be removed"
    kept = set(result.kept_indices)
    assert 100 not in kept, "the future-stamped duplicate must be the sample removed"
    assert 101 in kept, "the sample immediately after the spike must survive (prefix-max regression)"
    assert set(range(101, 107)).issubset(kept), "all six shadowed samples must survive"
    assert 107 in kept, "the legitimate packet carrying the duplicated stamp must survive"
    _assert_subsequence(result.kept_indices, values)


def test_removal_count_equals_the_theoretical_minimum_on_random_series():
    rng = random.Random(20260728)
    for _ in range(40):
        n = rng.randint(2, 120)
        values = [round(0.01 * i, 6) for i in range(n)]
        for _ in range(rng.randint(0, 1 + n // 8)):
            kind = rng.random()
            i = rng.randrange(n)
            if kind < 0.5:  # emit a packet early carrying a later packet's stamp
                values[i] = values[min(n - 1, i + rng.randint(1, 8))]
            elif kind < 0.8:  # swap a neighbouring pair
                j = min(n - 1, i + 1)
                values[i], values[j] = values[j], values[i]
            else:  # scramble a short burst in place
                lo, hi = i, min(n, i + rng.randint(2, 10))
                burst = values[lo:hi]
                rng.shuffle(burst)
                values[lo:hi] = burst
        result = repair(values, max_gap_s=1e9)
        expected = _reference_lis_length(values)
        assert len(result.kept_indices) == expected
        assert result.dropped_count == n - expected
        _assert_subsequence(result.kept_indices, values)


def test_increasing_subsequence_is_strict_not_merely_nondecreasing():
    values = [0.0, 1.0, 1.0, 1.0, 2.0]
    kept = increasing_subsequence(values)
    assert len(kept) == 3
    _assert_subsequence(kept, values)


# --- gap splitting ---------------------------------------------------------------------


def test_split_at_gaps_returns_contiguous_half_open_ranges():
    values = [0.0, 0.01, 0.02, 5.0, 5.01]
    assert split_at_gaps(values, 0.2) == [(0, 3), (3, 5)]
    assert split_at_gaps(values, 10.0) == [(0, 5)]
    assert split_at_gaps([1.0], 0.2) == [(0, 1)]


def test_gap_split_keeps_the_longest_segment():
    short = [round(0.01 * i, 6) for i in range(50)]          # 0.49 s
    long = [round(100.0 + 0.01 * i, 6) for i in range(300)]  # 2.99 s, after a 99.5 s outage
    tail = [round(500.0 + 0.01 * i, 6) for i in range(20)]   # 0.19 s
    values = short + long + tail
    result = repair(values)

    assert result.gap_count == 2
    assert result.segment_start == 50
    assert result.segment_end == 349
    assert len(result.kept_indices) == 300
    assert result.duration_s == pytest.approx(2.99)


def test_gap_split_measures_gaps_after_repair_not_before():
    # Before ordering repair the stray sample looks like a 1 s jump followed by a jump back.
    values = [0.0, 0.01, 1.0, 0.02, 0.03, 0.04]
    result = repair(values)
    assert result.gap_count == 0
    assert list(result.kept_indices) == [0, 1, 3, 4, 5]


@pytest.mark.parametrize(
    "values",
    [
        [0.0],
        [0.0, 0.01, 0.02],
        [0.0, 0.02, 0.01, 0.03],
        [0.0, 0.0, 0.0, 0.01],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.0, 0.01, 9.0, 9.01, 9.02, 9.03],
    ],
)
def test_repaired_series_is_strictly_increasing(values):
    result = repair(values)
    _assert_subsequence(result.kept_indices, values)
    assert result.dropped_count == len(values) - _reference_lis_length(values)


def test_repair_rejects_an_empty_series():
    with pytest.raises(ValueError):
        repair([])


def test_default_gap_threshold_is_200ms():
    assert MAX_GAP_S == pytest.approx(0.2)


# --- writing a loader-compatible derived drive -----------------------------------------


def _stamp(i):
    return f"2011-09-26 13:12:{3 + i // 100:02d}.{(i % 100) * 10_000_000:09d}"


def _packet(i):
    """A well-formed 30-column OXTS packet, distinct per index so byte-identity can bite."""
    return f"49.0{i:06d} 8.4 113.9 " + "0.1 " * 26 + "0.1\n"


def _write_source_drive(root, date, drive, stamps):
    oxts = root / date / f"{date}_drive_{drive}_extract" / "oxts"
    (oxts / "data").mkdir(parents=True)
    oxts.joinpath("timestamps.txt").write_text("".join(f"{s}\n" for s in stamps), encoding="utf-8")
    oxts.joinpath("dataformat.txt").write_text("lat: latitude\n", encoding="utf-8")
    for i in range(len(stamps)):
        oxts.joinpath("data", f"{i:010d}.txt").write_text(_packet(i), encoding="utf-8")
    return oxts


def _spiked_stamps(n=120, spike_at=40, duplicates=47):
    stamps = [_stamp(i) for i in range(n)]
    stamps[spike_at] = stamps[duplicates]
    return stamps


def test_write_clean_drive_produces_a_tree_the_frozen_loader_accepts(tmp_path):
    from kitti_highrate_loader import HighRateOxtsConfig, load_highrate_oxts

    date = "2011_09_26"
    _write_source_drive(tmp_path, date, "0015", _spiked_stamps())

    result = write_clean_drive(tmp_path, date, "0015")
    assert result.dropped_count == 1

    clean = tmp_path / date / f"{date}_drive_0015_clean_extract" / "oxts"
    assert clean.joinpath("timestamps.txt").is_file()
    assert clean.joinpath("dataformat.txt").is_file()
    assert len(list(clean.joinpath("data").glob("*.txt"))) == 119

    sequence = load_highrate_oxts(
        HighRateOxtsConfig(root=tmp_path, date=date, drive="0015_clean", cache_root=tmp_path / "cache")
    )
    assert sequence.sample_count == 119
    assert sequence.drive == "0015_clean"
    assert sequence.duration_s == pytest.approx(1.19, abs=1e-6)


def test_written_timestamps_are_a_byte_exact_subsequence_of_the_source(tmp_path):
    date = "2011_09_26"
    source = _write_source_drive(tmp_path, date, "0015", _spiked_stamps())

    write_clean_drive(tmp_path, date, "0015")
    clean = tmp_path / date / f"{date}_drive_0015_clean_extract" / "oxts"

    written = clean.joinpath("timestamps.txt").read_text(encoding="utf-8").splitlines()
    original = source.joinpath("timestamps.txt").read_text(encoding="utf-8").splitlines()
    assert written == [line for i, line in enumerate(original) if i != 40]


def test_written_packets_are_renumbered_from_zero_and_byte_identical_to_their_source(tmp_path):
    date = "2011_09_26"
    source = _write_source_drive(tmp_path, date, "0015", _spiked_stamps())

    result = write_clean_drive(tmp_path, date, "0015")
    clean = tmp_path / date / f"{date}_drive_0015_clean_extract" / "oxts" / "data"

    names = sorted(path.name for path in clean.glob("*.txt"))
    assert names == [f"{i:010d}.txt" for i in range(119)]
    for position, source_index in enumerate(result.kept_indices):
        written = hashlib.sha256(clean.joinpath(f"{position:010d}.txt").read_bytes()).hexdigest()
        expected = hashlib.sha256(
            source.joinpath("data", f"{source_index:010d}.txt").read_bytes()
        ).hexdigest()
        assert written == expected, f"packet {position} is not a byte copy of source {source_index}"


def test_write_clean_drive_rewrites_a_stale_derived_tree(tmp_path):
    date = "2011_09_26"
    _write_source_drive(tmp_path, date, "0015", _spiked_stamps())
    clean_data = tmp_path / date / f"{date}_drive_0015_clean_extract" / "oxts" / "data"
    clean_data.mkdir(parents=True)
    clean_data.joinpath("0000009999.txt").write_text("stale\n", encoding="utf-8")

    write_clean_drive(tmp_path, date, "0015")
    assert not clean_data.joinpath("0000009999.txt").exists(), "stale packets must not survive"
    assert len(list(clean_data.glob("*.txt"))) == 119


def test_write_clean_drive_keeps_only_the_longest_segment(tmp_path):
    date = "2011_09_26"
    # 30 samples, a 1 s outage, then 80 samples: the second segment must be the one written.
    stamps = [_stamp(i) for i in range(30)] + [_stamp(i) for i in range(130, 210)]
    _write_source_drive(tmp_path, date, "0015", stamps)

    result = write_clean_drive(tmp_path, date, "0015")
    assert result.gap_count == 1
    assert result.segment_start == 30
    assert result.segment_end == 109
    assert len(result.kept_indices) == 80

    clean = tmp_path / date / f"{date}_drive_0015_clean_extract" / "oxts"
    assert len(clean.joinpath("timestamps.txt").read_text(encoding="utf-8").splitlines()) == 80


# --- segment tiebreak: count first, duration second -------------------------------------


def test_longest_segment_is_chosen_by_sample_count_not_elapsed_time():
    # A sparse-but-long segment must not beat a dense shorter one: OXTS is a fixed-rate
    # sensor, so samples are the resource an estimator actually consumes.
    sparse = [round(0.15 * i, 6) for i in range(8)]           # 1.05 s, 8 samples
    dense = [round(11.0 + 0.01 * i, 6) for i in range(60)]    # 0.59 s, 60 samples
    result = repair(sparse + dense)

    assert result.gap_count == 1
    assert result.segment_start == 8, "the denser segment must win"
    assert result.segment_end == 67
    assert result.kept_count == 60
    assert result.duration_s == pytest.approx(0.59)


def test_equal_sample_counts_fall_back_to_the_longer_segment():
    sparse = [round(0.15 * i, 6) for i in range(20)]          # 2.85 s, 20 samples
    dense = [round(11.0 + 0.01 * i, 6) for i in range(20)]    # 0.19 s, 20 samples
    result = repair(sparse + dense)

    assert result.kept_count == 20
    assert result.segment_start == 0, "on a count tie the longer-elapsed segment wins"
    assert result.duration_s == pytest.approx(2.85)


# --- reporting: retention and residual intra-segment gaps --------------------------------


def test_intra_segment_gap_report_threshold_is_50ms():
    assert INTRA_SEGMENT_GAP_REPORT_S == pytest.approx(0.05)


def test_repair_reports_gaps_that_survive_inside_the_kept_segment():
    # 150 ms is under the 200 ms split threshold, so this hole stays in the kept segment.
    values = [round(0.01 * i, 6) for i in range(50)]
    values += [round(values[-1] + 0.15 + 0.01 * i, 6) for i in range(50)]
    result = repair(values)

    assert result.gap_count == 0, "a sub-threshold hole must not split the series"
    assert result.kept_count == 100
    assert result.max_intra_segment_gap_s == pytest.approx(0.15)
    assert result.intra_segment_gaps_over_50ms == 1


def test_a_clean_segment_reports_no_residual_gaps():
    result = repair([round(0.01 * i, 6) for i in range(200)])
    assert result.max_intra_segment_gap_s == pytest.approx(0.01)
    assert result.intra_segment_gaps_over_50ms == 0


def test_retention_counts_segment_selection_not_just_ordering_removals():
    short = [round(0.01 * i, 6) for i in range(40)]
    long = [round(100.0 + 0.01 * i, 6) for i in range(60)]
    result = repair(short + long)

    assert result.dropped_count == 0, "nothing is out of order here"
    assert result.reorder_drop_percent == pytest.approx(0.0)
    assert result.kept_count == 60
    assert result.retention_percent == pytest.approx(60.0), (
        "retention must account for the 40 samples segment selection discarded"
    )


def test_duration_original_s_spans_the_whole_source_series():
    short = [round(0.01 * i, 6) for i in range(40)]
    long = [round(100.0 + 0.01 * i, 6) for i in range(60)]
    result = repair(short + long)

    assert result.duration_s == pytest.approx(0.59)
    assert result.duration_original_s == pytest.approx(100.59)


# --- the independent longest-increasing-subsequence DP -----------------------------------


@pytest.mark.parametrize(
    ("values", "length"),
    [
        ([], 0),
        ([1.0], 1),
        ([0.0, 0.01, 0.02, 0.03], 4),
        ([0.0, 0.0, 0.0, 0.0], 1),
        ([0.4, 0.3, 0.2, 0.1], 1),
        ([0.0, 0.02, 0.01, 0.03], 3),
    ],
)
def test_longest_increasing_length_matches_hand_worked_cases(values, length):
    assert longest_increasing_length(values) == length
    assert minimum_removals(values) == len(values) - length


def test_minimum_removals_matches_an_independent_dp_on_random_series():
    rng = random.Random(4242)
    for _ in range(60):
        n = rng.randint(0, 80)
        values = [round(rng.uniform(0.0, 1.0), 6) for _ in range(n)]
        assert minimum_removals(values) == n - _reference_lis_length(values)


def test_minimum_removals_agrees_with_the_repair_it_guards():
    rng = random.Random(99)
    for _ in range(30):
        n = rng.randint(2, 90)
        values = [round(0.01 * i, 6) for i in range(n)]
        for _ in range(rng.randint(0, 6)):
            i = rng.randrange(n)
            values[i] = values[min(n - 1, i + rng.randint(1, 6))]
        assert repair(values, max_gap_s=1e9).dropped_count == minimum_removals(values)


# --- verify_clean_drive: it must actually catch corruption -------------------------------


def _written_drive(tmp_path, stamps=None):
    """Write a source drive and its derived tree; return (date, drive, result, clean oxts dir)."""
    date = "2011_09_26"
    _write_source_drive(tmp_path, date, "0015", _spiked_stamps() if stamps is None else stamps)
    result = write_clean_drive(tmp_path, date, "0015")
    clean = tmp_path / date / f"{date}_drive_0015_clean_extract" / "oxts"
    return date, "0015", result, clean


def test_verify_accepts_the_tree_write_clean_drive_just_produced(tmp_path):
    date, drive, result, _ = _written_drive(tmp_path)
    verify_clean_drive(tmp_path, date, drive, result)  # must not raise


def test_verify_accepts_a_drive_where_segment_selection_dropped_a_whole_run(tmp_path):
    # The window-maximality check must not fire on a legitimate multi-segment drive.
    stamps = [_stamp(i) for i in range(30)] + [_stamp(i) for i in range(130, 210)]
    date, drive, result, _ = _written_drive(tmp_path, stamps)
    assert result.segment_start == 30 and result.kept_count == 80
    verify_clean_drive(tmp_path, date, drive, result)  # must not raise


def test_verify_catches_a_flipped_byte_in_a_written_packet(tmp_path):
    date, drive, result, clean = _written_drive(tmp_path)
    victim = clean / "data" / "0000000007.txt"
    raw = bytearray(victim.read_bytes())
    raw[3] ^= 0x01
    victim.write_bytes(bytes(raw))

    with pytest.raises(AssertionError, match="byte copy"):
        verify_clean_drive(tmp_path, date, drive, result)


def test_verify_catches_an_edited_timestamp_line(tmp_path):
    date, drive, result, clean = _written_drive(tmp_path)
    lines = clean.joinpath("timestamps.txt").read_text(encoding="utf-8").splitlines()
    lines[5] = "2011-09-26 13:12:03.055000000"  # still ordered, still gap-free, not the source
    clean.joinpath("timestamps.txt").write_text(
        "".join(f"{line}\n" for line in lines), encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="byte-exact subsequence"):
        verify_clean_drive(tmp_path, date, drive, result)


def test_verify_catches_a_deleted_packet_file(tmp_path):
    date, drive, result, clean = _written_drive(tmp_path)
    clean.joinpath("data", "0000000050.txt").unlink()

    with pytest.raises(AssertionError, match="numbered contiguously"):
        verify_clean_drive(tmp_path, date, drive, result)


def test_verify_catches_an_inflated_dropped_count(tmp_path):
    date, drive, result, _ = _written_drive(tmp_path)
    with pytest.raises(AssertionError, match="theoretical minimum"):
        verify_clean_drive(tmp_path, date, drive, replace(result, dropped_count=result.dropped_count + 5))


def test_verify_catches_two_swapped_packets(tmp_path):
    date, drive, result, clean = _written_drive(tmp_path)
    first = clean / "data" / "0000000011.txt"
    second = clean / "data" / "0000000012.txt"
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    first.write_bytes(second_bytes)
    second.write_bytes(first_bytes)

    with pytest.raises(AssertionError, match="byte copy"):
        verify_clean_drive(tmp_path, date, drive, result)


def test_verify_catches_a_truncated_timestamps_file(tmp_path):
    date, drive, result, clean = _written_drive(tmp_path)
    lines = clean.joinpath("timestamps.txt").read_text(encoding="utf-8").splitlines()[:50]
    clean.joinpath("timestamps.txt").write_text(
        "".join(f"{line}\n" for line in lines), encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="byte-exact subsequence"):
        verify_clean_drive(tmp_path, date, drive, result)


def test_verify_catches_a_segment_that_is_not_maximal_in_its_own_index_window(tmp_path, monkeypatch):
    """dropped_count is a PRE-segment-selection number, so it cannot see this.

    A bug in split_at_gaps or the tiebreak that needlessly leaves out an interior sample keeps
    dropped_count at the true global minimum; only a check over the written window catches it.
    """
    real_repair = sanitize_oxts.repair

    def lossy(values, max_gap_s=MAX_GAP_S):
        result = real_repair(values, max_gap_s)
        return replace(result, kept_indices=tuple(i for i in result.kept_indices if i != 60))

    monkeypatch.setattr(sanitize_oxts, "repair", lossy)
    date, drive, result, _ = _written_drive(tmp_path)

    assert result.dropped_count == 1, "the pre-segment number is still the true minimum"
    assert result.kept_count == 118, "but one more sample than that was actually thrown away"
    with pytest.raises(AssertionError, match="maximal"):
        verify_clean_drive(tmp_path, date, drive, result)


def test_verify_skips_the_expensive_checks_when_asked(tmp_path):
    date, drive, result, _ = _written_drive(tmp_path)
    verify_clean_drive(tmp_path, date, drive, replace(result, dropped_count=99),
                       check_minimality=False)  # must not raise


# --- main() ------------------------------------------------------------------------------


def _run_main(tmp_path, drives, extra=()):
    root = tmp_path / "raw"
    report = tmp_path / "report.csv"
    argv = [
        "--root", str(root),
        "--cache-root", str(tmp_path / "cache"),
        "--report", str(report),
        "--drives", ",".join(f"2011_09_26:{drive}" for drive in drives),
        *extra,
    ]
    return main(argv), report


def test_main_writes_a_report_naming_what_each_number_measures(tmp_path, capsys):
    root = tmp_path / "raw"
    _write_source_drive(root, "2011_09_26", "0015", _spiked_stamps())
    _write_source_drive(root, "2011_09_26", "0042", [_stamp(i) for i in range(60)])

    code, report = _run_main(tmp_path, ["0015", "0042"])
    assert code == 0

    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(REPORT_COLUMNS)
    assert "n_dropped" not in rows[0], "the old ambiguous column name must be gone"
    assert "drop_percent" not in rows[0]

    spiked = rows[0]
    assert spiked["n_original"] == "120"
    assert spiked["n_reordered_drops"] == "1"
    assert spiked["reorder_drop_percent"] == "0.833"
    assert spiked["n_kept"] == "119"
    assert spiked["retention_percent"] == "99.167"
    assert spiked["duration_s"] == "1.190"
    assert spiked["duration_original_s"] == "1.190"
    assert spiked["max_intra_segment_gap_s"] == "0.020000"
    assert spiked["n_intra_segment_gaps_over_50ms"] == "0"

    clean = rows[1]
    assert clean["n_reordered_drops"] == "0"
    assert clean["retention_percent"] == "100.000"


def test_main_summary_reports_both_ordering_removals_and_true_retention(tmp_path, capsys):
    root = tmp_path / "raw"
    _write_source_drive(root, "2011_09_26", "0015", _spiked_stamps())
    # 30 samples, a 1 s outage, then 80: segment selection discards 30 of 110 here.
    _write_source_drive(root, "2011_09_26", "0009",
                        [_stamp(i) for i in range(30)] + [_stamp(i) for i in range(130, 210)])

    code, _ = _run_main(tmp_path, ["0015", "0009"])
    assert code == 0
    out = capsys.readouterr().out

    assert "ordering removals: 1 of 230" in out
    assert "199 of 230" in out, "the summary must state true overall retention"
    assert "86.522%" in out
    assert "31 discarded" in out
    assert "0009_clean" in out.split("retaining under")[-1], (
        "a drive under 95% retention must be named"
    )
    assert "72.727" in out


def test_main_names_no_drive_when_every_drive_is_above_the_retention_floor(tmp_path, capsys):
    root = tmp_path / "raw"
    _write_source_drive(root, "2011_09_26", "0015", _spiked_stamps())

    assert _run_main(tmp_path, ["0015"])[0] == 0
    assert "retaining under" not in capsys.readouterr().out


def test_main_rejects_a_malformed_drives_spec():
    with pytest.raises(ValueError, match="date:drive"):
        main(["--drives", "2011_09_26"])
    with pytest.raises(ValueError, match="selected no drives"):
        main(["--drives", " , "])


def test_main_load_check_bypasses_the_cache_rather_than_deleting_it(tmp_path, monkeypatch):
    """The load check must not rest on the script reaching into the cache directory."""
    assert not hasattr(sanitize_oxts, "cache_path_for"), (
        "the script should no longer need to know where cache files live"
    )

    root = tmp_path / "raw"
    _write_source_drive(root, "2011_09_26", "0015", _spiked_stamps())

    seen = []
    real_load = sanitize_oxts.load_highrate_oxts
    monkeypatch.setattr(sanitize_oxts, "load_highrate_oxts",
                        lambda config: (seen.append(config), real_load(config))[1])

    assert _run_main(tmp_path, ["0015"])[0] == 0
    assert seen, "the load check must actually load something"
    assert all(config.force_refresh for config in seen), (
        "every load-check read must bypass the cache"
    )


def test_main_load_check_validates_the_tree_not_a_poisoned_cache(tmp_path, monkeypatch):
    from kitti_highrate_loader import (  # noqa: E402
        HighRateOxtsConfig,
        cache_path_for,
        load_cache,
        load_highrate_oxts,
        save_cache,
    )

    root = tmp_path / "raw"
    _write_source_drive(root, "2011_09_26", "0015", _spiked_stamps())
    assert _run_main(tmp_path, ["0015"])[0] == 0

    config = HighRateOxtsConfig(root=root, date="2011_09_26", drive="0015_clean",
                                cache_root=tmp_path / "cache")
    path = cache_path_for(config)
    good = load_cache(path)
    poisoned = replace(good, **{
        name: getattr(good, name)[:10] for name in (
            "timestamps", "lat_lon_alt", "enu_position_m", "roll_pitch_yaw",
            "velocity", "accel_body", "gyro_body",
        )
    })
    save_cache(poisoned, path)
    assert load_highrate_oxts(config).sample_count == 10, "the poisoned cache must really load"

    # Neutralise the old defence: if the load check only worked because the script deleted the
    # cache first, this run reads the poisoned 10-sample sequence and the count assertion trips.
    monkeypatch.setattr(pathlib.Path, "unlink", lambda self, **kwargs: None)
    assert _run_main(tmp_path, ["0015"])[0] == 0
    assert load_highrate_oxts(config).sample_count == 119, "the cache must be rewritten correctly"
