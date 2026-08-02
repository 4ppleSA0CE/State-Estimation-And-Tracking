#!/usr/bin/env python3
"""Fetch only the oxts/ tree out of remote KITTI raw zips using HTTP range requests.

A KITTI extract zip is 2-9 GB, almost all of it images and Velodyne scans. The ESKF needs
oxts/ alone (~1.4 MB). S3 serves ranges, so a seekable file object over Range: bytes=
lets stdlib zipfile read the central directory and inflate individual members without
downloading the archive. Measured on drive_0015: 1.82 GB remote -> 5.88 MB pulled, 2 requests.

The block cache is what makes it 2 requests instead of 3128: without it every member read is
its own round trip.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BLOCK = 4 << 20              # 4 MiB read-through blocks
# Per-archive budget, enforced inside _block as bytes arrive. The oxts tree of the largest drive
# costs ~25 MB, so 128 MB is generous headroom while still being orders of magnitude below the
# 9.7 GB an ignored Range header would deliver. A global end-of-run total cannot serve this
# purpose: by the time it is evaluated the bytes are already in memory and on disk, and on a warm
# run every drive is skipped so the total checks nothing at all.
MAX_ARCHIVE_BYTES = 128 << 20
BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

# (date, drive) pairs to fetch: all eight the validation sweep evaluates.
#
# 0001 and 0009 were already on the development machine, but `data/` is gitignored, so a fresh
# clone has neither -- omitting them here made the README's reproduce-the-results sequence fail
# at the first step for everyone except the author. Fetching is idempotent (a drive whose
# oxts/timestamps.txt exists is skipped), so listing them costs an already-populated tree nothing.
DRIVES = [
    ("2011_09_26", "0001"), ("2011_09_26", "0009"),
    ("2011_09_26", "0015"), ("2011_09_26", "0117"),
    ("2011_09_30", "0020"), ("2011_09_30", "0033"),
    ("2011_10_03", "0042"), ("2011_09_29", "0004"),
]

AB3DMOT_RAW = ("https://raw.githubusercontent.com/xinshuoweng/AB3DMOT/master/"
               "data/KITTI/detection/pointrcnn_Car_val")
VAL_SEQUENCES = ["0001", "0006", "0008", "0010", "0012", "0013",
                 "0014", "0015", "0016", "0018", "0019"]


class _UrllibOpener:
    def open(self, req):
        return urllib.request.urlopen(req)

    def head_size_for(self, url):
        return int(urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD")).headers["Content-Length"])


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with 4 MiB read-through blocks."""

    def __init__(self, url, opener=None, block_size=BLOCK, budget_bytes=MAX_ARCHIVE_BYTES):
        self.budget_bytes = budget_bytes
        self.url = url
        self._opener = opener or _UrllibOpener()
        self._pos = 0
        self._block_size = block_size
        self._blocks: dict[int, bytes] = {}
        self.bytes_pulled = 0
        self.requests = 0
        self.size = (self._opener.head_size() if hasattr(self._opener, "head_size")
                     else self._opener.head_size_for(url))

    @property
    def fetched_blocks(self):
        return set(self._blocks)

    def seek(self, off, whence=0):
        base = {0: 0, 1: self._pos, 2: self.size}[whence]
        self._pos = max(0, base + off)
        return self._pos

    def tell(self): return self._pos
    def seekable(self): return True
    def readable(self): return True

    def _block(self, idx):
        if idx not in self._blocks:
            start = idx * self._block_size
            end = min(start + self._block_size, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            resp = self._opener.open(req)
            # A server that ignores Range answers 200 with the whole archive; that is the
            # silent degradation this fetcher exists to avoid, so refuse it before read().
            status = getattr(resp, "status", None)
            if status is not None and status != 206:
                raise RuntimeError(
                    f"{self.url}: expected 206 for range {start}-{end}, got {status} "
                    "(server ignored the range request)")
            # Bounded read: a server that ignores Range and streams the whole archive must not
            # be able to pull 9.7 GB into memory before any check runs. One extra byte is
            # requested so an over-long body is detectable rather than silently truncated.
            expected = end - start + 1
            data = resp.read(expected + 1)
            self.requests += 1
            self.bytes_pulled += len(data)
            if len(data) != expected:
                raise RuntimeError(
                    f"{self.url}: range {start}-{end} returned {len(data)} bytes, "
                    f"expected {expected} (server ignored the range request)")
            # Budget enforced HERE, as blocks are pulled -- not as a verdict after the fact.
            # A ceiling that only reports at the end cannot prevent anything.
            if self.budget_bytes is not None and self.bytes_pulled > self.budget_bytes:
                raise RuntimeError(
                    f"{self.url}: pulled {self.bytes_pulled/1e6:.1f} MB, over the "
                    f"{self.budget_bytes/1e6:.0f} MB budget for a single archive -- range "
                    "requests are not being honoured as expected")
            self._blocks[idx] = data
        return self._blocks[idx]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        out = bytearray()
        while n > 0 and self._pos < self.size:
            idx, off = divmod(self._pos, self._block_size)
            chunk = self._block(idx)[off:off + n]
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def oxts_members(zf: zipfile.ZipFile) -> list[str]:
    return [n for n in zf.namelist() if "/oxts/" in n and not n.endswith("/")]


def extract_oxts(zf: zipfile.ZipFile, dest: Path) -> list[Path]:
    """Write every oxts/ member under `dest`, refusing any name that escapes it.

    The real KITTI archives contain no hostile names, but `dest / name` follows pathlib's
    absolute-path semantics -- an absolute member name would discard `dest` entirely -- and `..`
    is not otherwise blocked. The guard costs one comparison per member.
    """
    dest = Path(dest).resolve()
    written = []
    for name in oxts_members(zf):
        out = (dest / name).resolve()
        if not out.is_relative_to(dest):
            raise RuntimeError(f"refusing zip member that escapes the destination: {name!r}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zf.read(name))
        written.append(out)
    return written


def drive_url(date: str, drive: str) -> str:
    stem = f"{date}_drive_{drive}"
    return f"{BASE}/raw_data/{stem}/{stem}_extract.zip"


def fetch_drive(date: str, drive: str, raw_root: Path) -> int:
    """Fetch one drive's oxts/ tree, publishing it only once it is complete.

    Extraction goes to a temporary directory and is moved into place with a single atomic
    rename. Writing straight to the destination would make an interrupted fetch permanently
    indistinguishable from a finished one: `timestamps.txt` is the FIRST member written, so the
    presence marker would already exist over a half-written tree, every later run would skip it,
    and a silently truncated trajectory would reach the filter.
    """
    stem = f"{date}_drive_{drive}_extract"
    final = raw_root / date / stem
    marker = final / "oxts" / "timestamps.txt"
    if marker.exists():
        print(f"  {stem}: already present, skipping")
        return 0

    raw_root.mkdir(parents=True, exist_ok=True)
    f = HttpFile(drive_url(date, drive))
    with tempfile.TemporaryDirectory(dir=raw_root, prefix=f".{stem}.partial-") as tmp:
        with zipfile.ZipFile(f) as zf:
            written = extract_oxts(zf, Path(tmp))
        staged = Path(tmp) / date / stem
        if not (staged / "oxts" / "timestamps.txt").exists():
            raise RuntimeError(f"{stem}: extraction produced no oxts/timestamps.txt")
        if final.exists():
            # Only reachable when a previous run died mid-extraction under the old scheme:
            # the directory exists but has no marker, so it is known-incomplete.
            print(f"  {stem}: discarding an incomplete earlier extraction")
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, final)
    print(f"  {stem}: {len(written)} files, {f.bytes_pulled/1e6:.2f} MB pulled, "
          f"{f.requests} requests, remote size {f.size/1e9:.2f} GB")
    return f.bytes_pulled


def fetch_tracking(root: Path) -> int:
    """KITTI Tracking GT labels + AB3DMOT PointRCNN Car detections for the val split."""
    pulled = 0
    label_dir = root / "training" / "label_02"
    if not (label_dir / "0001.txt").exists():
        url = f"{BASE}/data_tracking_label_2.zip"
        f = HttpFile(url)
        zf = zipfile.ZipFile(f)
        for name in zf.namelist():
            if name.endswith(".txt"):
                out = root / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(name))
        pulled += f.bytes_pulled
        print(f"  labels: {f.bytes_pulled/1e6:.2f} MB pulled, {f.requests} requests")
    else:
        print("  labels: already present, skipping")

    det_dir = root / "ab3dmot_car_val"
    det_dir.mkdir(parents=True, exist_ok=True)
    for seq in VAL_SEQUENCES:
        out = det_dir / f"{seq}.txt"
        if out.exists():
            continue
        data = urllib.request.urlopen(f"{AB3DMOT_RAW}/{seq}.txt").read()
        out.write_bytes(data)
        pulled += len(data)
    print(f"  detections: {len(VAL_SEQUENCES)} val sequences in {det_dir}")
    return pulled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=Path("data/kitti_raw"))
    ap.add_argument("--tracking-root", type=Path, default=Path("data/kitti_tracking"))
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-tracking", action="store_true")
    args = ap.parse_args(argv)

    total = 0
    if not args.skip_raw:
        print("raw drives:")
        for date, drive in DRIVES:
            total += fetch_drive(date, drive, args.raw_root)
    if not args.skip_tracking:
        print("tracking:")
        total += fetch_tracking(args.tracking_root)

    # No end-of-run ceiling here on purpose. The real limit is MAX_ARCHIVE_BYTES, enforced per
    # archive inside HttpFile._block as bytes arrive, which can actually stop a runaway download.
    # A total checked at this point would already be too late, and on a warm run where every
    # drive is skipped it would be trivially satisfied while checking nothing.
    print(f"\ntotal pulled: {total/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
