#!/usr/bin/env python3
"""Merge two (or more) parsed datasets into a single one.

Each input directory is expected to be a folder produced by ``parse.py``
(i.e. it contains the CSVs listed in ``clean_directory/dataset_layout.yaml``
plus a ``used_rosbags.txt`` sidecar).  This script concatenates the CSVs
end-to-end on a single timeline, inserting a configurable buffer (default
10 s, matching ``parse.BUFFER_NS``) between consecutive datasets and
shifting all timestamps of subsequent datasets accordingly.  The merged
``used_rosbags.txt`` is rewritten with the combined bag list, the new
join times, and the new total duration.

Usage:
    python3 merge_datasets.py --inputs A_dir B_dir [C_dir ...] \\
                              --output merged_dir [--buffer 10000000000]
"""

import argparse
import csv
import os
import sys

from clean_directory.dataset_layout import LAYOUT as _LAYOUT


BUFFER_NS = 10 * 10**9  # default 10 s gap, matches parse.BUFFER_NS


# ============================================================================
# CSV helpers
# ============================================================================

def _csv_max_t(path):
    """Return the largest integer timestamp in column ``t`` of ``path``.

    Returns ``None`` if the file is missing or contains no data rows.
    """
    if not os.path.exists(path):
        return None
    max_t = None
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        if "t" not in (r.fieldnames or []):
            return None
        for row in r:
            try:
                t = int(row["t"])
            except (TypeError, ValueError):
                continue
            if max_t is None or t > max_t:
                max_t = t
    return max_t


def _append_csv_with_offset(src_path, dst_path, t_offset):
    """Append ``src_path`` to ``dst_path``, shifting the ``t`` column.

    If ``dst_path`` does not exist yet, the header from ``src_path`` is
    copied first.  If ``dst_path`` already exists, the source header is
    skipped and the rows are appended with their ``t`` value increased
    by ``t_offset``.
    """
    if not os.path.exists(src_path):
        print(f"  skip (missing): {os.path.basename(src_path)}")
        return 0

    new_file = not os.path.exists(dst_path)
    n_rows = 0
    with open(src_path, "r", newline="") as fi, \
         open(dst_path, "a", newline="") as fo:
        reader = csv.reader(fi)
        writer = csv.writer(fo)

        header = next(reader, None)
        if header is None:
            return 0
        if "t" not in header:
            raise ValueError(
                f"{src_path} has no 't' column (header={header})")
        t_idx = header.index("t")

        if new_file:
            writer.writerow(header)

        for row in reader:
            if not row:
                continue
            try:
                row[t_idx] = str(int(row[t_idx]) + t_offset)
            except ValueError:
                # Leave non-integer timestamps untouched (shouldn't happen
                # in well-formed parse.py output, but don't crash).
                pass
            writer.writerow(row)
            n_rows += 1

    return n_rows


# ============================================================================
# Metadata helpers
# ============================================================================

def _read_metadata(path):
    """Parse a ``used_rosbags.txt`` file written by ``parse.py``.

    Returns ``(bag_paths, join_times_ns, total_ns)``.  Missing or
    malformed fields fall back to sensible defaults so older sidecars
    still work.
    """
    bag_paths = []
    join_times = []
    total_ns = 0

    if not os.path.exists(path):
        return bag_paths, join_times, total_ns

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Join times:"):
                payload = line.split(":", 1)[1].strip()
                if payload:
                    for tok in payload.split(","):
                        tok = tok.strip()
                        if tok:
                            try:
                                join_times.append(int(tok))
                            except ValueError:
                                pass
            elif line.startswith("Total hours:"):
                try:
                    hours = float(line.split(":", 1)[1].strip())
                    total_ns = int(round(hours * 3600e9))
                except ValueError:
                    pass
            else:
                bag_paths.append(line)

    return bag_paths, join_times, total_ns


def _write_metadata(out_dir, bag_paths, join_times_ns, total_ns):
    path = os.path.join(out_dir, _LAYOUT["used_rosbags"])
    with open(path, "w") as f:
        for p in bag_paths:
            f.write(f"{p}\n")
        f.write("\n")
        f.write("Join times: " + ",".join(str(j) for j in join_times_ns) + "\n")
        f.write("\n")
        f.write(f"Total hours: {total_ns / 3600e9:.2f}\n")


# ============================================================================
# Per-dataset processing
# ============================================================================

# All LAYOUT entries that are CSVs with a 't' column (i.e. everything
# except the metadata sidecar).
_CSV_KEYS = [k for k in _LAYOUT.keys() if k != "used_rosbags"]


def _dataset_max_t(in_dir):
    """Largest timestamp across every known CSV in ``in_dir``."""
    max_t = None
    for key in _CSV_KEYS:
        t = _csv_max_t(os.path.join(in_dir, _LAYOUT[key]))
        if t is None:
            continue
        if max_t is None or t > max_t:
            max_t = t
    return max_t or 0


def _append_dataset(in_dir, out_dir, t_offset):
    """Append every known CSV from ``in_dir`` to ``out_dir`` with offset."""
    for key in _CSV_KEYS:
        src = os.path.join(in_dir, _LAYOUT[key])
        dst = os.path.join(out_dir, _LAYOUT[key])
        n = _append_csv_with_offset(src, dst, t_offset)
        print(f"  {_LAYOUT[key]:<40s}  +{n} rows")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Concatenate two or more parsed datasets onto a single "
                    "timeline with a buffer gap.")
    ap.add_argument("--inputs", required=True, nargs="+",
                    help="Two or more parsed-dataset directories, in the "
                         "order they should appear on the merged timeline.")
    ap.add_argument("--output", required=True,
                    help="Destination directory (created if needed). Must "
                         "not be one of the inputs.")
    ap.add_argument("--buffer", type=int, default=BUFFER_NS,
                    help=f"Gap between datasets in ns (default {BUFFER_NS}).")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("--inputs needs at least two directories")

    inputs = [os.path.abspath(p) for p in args.inputs]
    out_dir = os.path.abspath(args.output)

    for p in inputs:
        if not os.path.isdir(p):
            ap.error(f"input not a directory: {p}")
        if os.path.abspath(p) == out_dir:
            ap.error("--output must not be one of --inputs")

    os.makedirs(out_dir, exist_ok=True)

    # Refuse to overwrite anything pre-existing in the output dir to avoid
    # silently appending to leftover CSVs from a previous merge.
    for key in _CSV_KEYS:
        dst = os.path.join(out_dir, _LAYOUT[key])
        if os.path.exists(dst):
            sys.exit(f"refusing to overwrite existing file: {dst}\n"
                     f"(remove it or pick an empty --output directory)")

    print(f"Merging {len(inputs)} datasets into {out_dir}")
    print(f"Buffer: {args.buffer} ns ({args.buffer / 1e9:.2f} s)")

    timeline_end = 0
    join_times = []
    bag_paths_all = []

    for idx, in_dir in enumerate(inputs):
        print(f"\n[{idx + 1}/{len(inputs)}] {in_dir}")

        if idx == 0:
            t_offset = 0
        else:
            t_offset = timeline_end + args.buffer
            join_times.append(t_offset)
        print(f"  t_offset = {t_offset / 1e9:.2f} s")

        _append_dataset(in_dir, out_dir, t_offset)

        ds_max_t = _dataset_max_t(in_dir)
        timeline_end = t_offset + ds_max_t
        print(f"  dataset duration ~ {ds_max_t / 1e9:.2f} s, "
              f"timeline_end = {timeline_end / 1e9:.2f} s")

        bag_paths_in, _, _ = _read_metadata(
            os.path.join(in_dir, _LAYOUT["used_rosbags"]))
        bag_paths_all.extend(bag_paths_in)

    _write_metadata(out_dir, bag_paths_all, join_times, timeline_end)
    print(f"\nDone. Total: {timeline_end / 3600e9:.2f} h. Wrote {out_dir}")


if __name__ == "__main__":
    main()
