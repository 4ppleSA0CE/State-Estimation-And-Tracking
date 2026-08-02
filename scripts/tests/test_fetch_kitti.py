"""Tests for the range-request KITTI fetcher. No network: ranges are served from memory."""
import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch_kitti import HttpFile, extract_oxts, oxts_members  # noqa: E402

BIG = "d/image_00/data/big.bin"
BIG_BYTES = 200_000
TEST_BLOCK = 4096  # small blocks so the fake archive spans many of them


class FakeOpener:
    """Serves Range requests out of an in-memory blob and counts requests."""

    def __init__(self, blob, ignore_ranges=False):
        self.blob = blob
        self.requests = 0
        self.ranges: list[tuple[int, int]] = []
        self.ignore_ranges = ignore_ranges

    def open(self, req):
        self.requests += 1
        rng = req.headers.get("Range")
        if rng is None or self.ignore_ranges:
            return _FakeResponse(self.blob, status=200)
        start, end = rng.split("=")[1].split("-")
        start, end = int(start), int(end)
        self.ranges.append((start, end))
        return _FakeResponse(self.blob[start:end + 1], status=206)

    def head_size(self):
        return len(self.blob)


class _FakeResponse(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status


def _make_zip(n_members=50):
    """oxts members first, then one large STORED member that must never be pulled."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("d/oxts/timestamps.txt", "2011-09-26 13:12:03.538669617\n" * n_members)
        for i in range(n_members):
            z.writestr(f"d/oxts/data/{i:010d}.txt", "49.0 8.4 113.9 " + "0.1 " * 27 + "\n")
        # STORED, not DEFLATED: deflate would squash 200 kB of zeros to ~200 bytes and the
        # "we never pulled the image payload" assertions below would pass vacuously.
        z.writestr(BIG, b"\x00" * BIG_BYTES, zipfile.ZIP_STORED)
    return buf.getvalue()


def _http_file(blob, **kw):
    return HttpFile("http://x/y.zip", opener=FakeOpener(blob, **kw), block_size=TEST_BLOCK)


def test_reads_all_oxts_members_in_few_requests():
    blob = _make_zip()
    f = _http_file(blob)
    zf = zipfile.ZipFile(f)
    names = oxts_members(zf)
    assert len(names) == 51  # 50 data files + timestamps.txt
    total = sum(len(zf.read(n)) for n in names)
    assert total > 0
    n_blocks = -(-len(blob) // TEST_BLOCK)
    assert f.requests < n_blocks // 4, f"block cache not working: {f.requests} of {n_blocks} blocks"


def test_repeated_reads_hit_the_block_cache():
    blob = _make_zip()
    f = _http_file(blob)
    zf = zipfile.ZipFile(f)
    names = oxts_members(zf)
    for n in names:
        zf.read(n)
    first = f.requests
    for n in names:
        zf.read(n)
    assert f.requests == first, "re-reading the same members issued new HTTP requests"


def test_never_reads_the_image_payload():
    blob = _make_zip()
    f = _http_file(blob)
    zf = zipfile.ZipFile(f)
    big = zf.getinfo(BIG)
    for n in oxts_members(zf):
        zf.read(n)
    assert f.bytes_pulled < len(blob), "pulled the whole archive"
    assert f.bytes_pulled < len(blob) // 4, f"pulled {f.bytes_pulled} of {len(blob)} bytes"
    # No block wholly inside the big member's payload may have been fetched.
    lo = big.header_offset + TEST_BLOCK  # skip the block straddling the oxts/big boundary
    hi = big.header_offset + BIG_BYTES - TEST_BLOCK
    interior = {i for i in f.fetched_blocks if lo <= i * TEST_BLOCK < hi}
    assert not interior, f"fetched blocks inside the image payload: {sorted(interior)}"


def test_extract_writes_expected_tree(tmp_path):
    blob = _make_zip(3)
    f = _http_file(blob)
    written = extract_oxts(zipfile.ZipFile(f), tmp_path)
    assert (tmp_path / "d" / "oxts" / "timestamps.txt").exists()
    assert (tmp_path / "d" / "oxts" / "data" / "0000000000.txt").exists()
    assert not (tmp_path / "d" / "image_00").exists()
    assert len(written) == 4


def test_seek_whence_modes():
    blob = _make_zip(2)
    f = _http_file(blob)
    assert f.seek(0, 2) == len(blob)
    assert f.seek(-10, 2) == len(blob) - 10
    f.seek(5)
    assert f.seek(3, 1) == 8


def test_read_matches_the_underlying_bytes_across_block_boundaries():
    blob = _make_zip(2)
    f = _http_file(blob)
    f.seek(TEST_BLOCK - 7)
    assert f.read(TEST_BLOCK + 21) == blob[TEST_BLOCK - 7:2 * TEST_BLOCK + 14]
    f.seek(len(blob) - 3)
    assert f.read(100) == blob[-3:]  # short read at EOF, not an error
    f.seek(0)
    assert f.read() == blob  # read(-1) reads to EOF


def test_server_ignoring_range_is_an_error_not_a_silent_full_download():
    blob = _make_zip(2)
    f = _http_file(blob, ignore_ranges=True)
    with pytest.raises(RuntimeError, match="range"):
        f.read(16)


# --- regression tests added after independent review -----------------------------------

def test_range_ignoring_server_is_refused_before_the_body_is_buffered():
    """The guard must PREVENT a full download, not merely report one afterwards.

    Review found that deleting the 206 status check still passed, because the length check
    catches the same case -- but only after `resp.read()` had already pulled the entire archive
    into memory. Bounding the read is the actual protection, so assert on how much was buffered.
    """
    blob = _make_zip(200)
    f = HttpFile("http://x/y.zip", opener=FakeOpener(blob, ignore_ranges=True), block_size=4096)
    with pytest.raises(RuntimeError, match="ignored the range request"):
        zipfile.ZipFile(f)
    assert f.bytes_pulled <= 4096 + 1, (
        f"buffered {f.bytes_pulled} bytes of a {len(blob)}-byte archive before refusing")


def test_per_archive_budget_stops_a_runaway_pull():
    """The byte budget must fire while blocks are arriving, not as an end-of-run verdict."""
    blob = _make_zip(400)
    f = HttpFile("http://x/y.zip", opener=FakeOpener(blob), block_size=1024, budget_bytes=4096)
    with pytest.raises(RuntimeError, match="budget"):
        zf = zipfile.ZipFile(f)
        for name in oxts_members(zf):
            zf.read(name)
    assert f.bytes_pulled <= 4096 + 1024


def test_seek_to_a_negative_absolute_position_is_clamped():
    """A negative _pos makes divmod yield a negative block index and a silently wrong read."""
    f = _http_file(_make_zip(2))
    assert f.seek(-10) == 0
    assert f.seek(5) == 5
    assert f.seek(-100, 1) == 0
    assert f.tell() == 0


def test_interrupted_extraction_does_not_leave_a_skippable_partial_tree(tmp_path, monkeypatch):
    """The presence marker must never appear over an incomplete tree.

    `oxts/timestamps.txt` is the FIRST member written, so writing straight to the destination
    made an aborted fetch permanently indistinguishable from a complete one: every later run
    printed "already present, skipping" and a truncated trajectory reached the filter.
    """
    import fetch_kitti

    blob = _make_zip(200)
    calls = {"n": 0}
    real_read = zipfile.ZipFile.read

    def exploding_read(self, name, *a, **kw):
        calls["n"] += 1
        if calls["n"] > 5:
            raise OSError("simulated interruption mid-extraction")
        return real_read(self, name, *a, **kw)

    monkeypatch.setattr(zipfile.ZipFile, "read", exploding_read)
    monkeypatch.setattr(fetch_kitti, "HttpFile",
                        lambda url, **kw: _http_file(blob, **kw))

    with pytest.raises(OSError, match="simulated interruption"):
        fetch_kitti.fetch_drive("d", "0000", tmp_path)

    marker = tmp_path / "d" / "d_drive_0000_extract" / "oxts" / "timestamps.txt"
    assert not marker.exists(), "presence marker published over an incomplete extraction"
    leftovers = [p for p in tmp_path.iterdir() if not p.name.startswith(".")]
    assert leftovers == [], f"partial tree left behind: {leftovers}"


class _LyingOpener(FakeOpener):
    """Answers 206 as required, then sends the whole archive anyway.

    The nastier case than a plain range-ignoring server: the status guard is satisfied, so the
    bounded read is the ONLY thing standing between us and buffering a 9.7 GB body.
    """

    def open(self, req):
        self.requests += 1
        return _FakeResponse(self.blob, status=206)


def test_206_with_an_oversized_body_is_still_bounded():
    blob = _make_zip(200)
    f = HttpFile("http://x/y.zip", opener=_LyingOpener(blob), block_size=4096)
    with pytest.raises(RuntimeError, match="ignored the range request"):
        zipfile.ZipFile(f)
    assert f.bytes_pulled <= 4096 + 1, (
        f"buffered {f.bytes_pulled} bytes of a {len(blob)}-byte body despite a 4096-byte range")
