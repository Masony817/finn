"""Line-prefixed telemetry CSV, e.g. finn's serial capture format.

The stream interleaves tagged lines; rows we care about start with a prefix
(default "data,") and the first prefixed line matching header_marker (or simply
the first prefixed line) is the CSV header. Non-numeric cells become NaN so
enum-ish columns (state names) do not break numeric loading.

The parser is line-oriented on purpose: the same tokenizer will serve the live
serial path later, one line at a time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scopik.datamodel import ScopikError
from scopik.profile import Profile


class PrefixedCsvSource:
    def read_table(
        self, path: Path, profile: Profile
    ) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        prefix = profile.source.line_prefix
        header_marker = profile.source.header_marker

        data_lines = [
            line
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith(prefix)
        ]
        if not data_lines:
            raise ScopikError(f"no {prefix!r}-prefixed lines in {path}")

        if header_marker is not None:
            header_index = next(
                (i for i, line in enumerate(data_lines) if line.startswith(header_marker)),
                None,
            )
            if header_index is None:
                raise ScopikError(f"no header line starting with {header_marker!r} in {path}")
        else:
            header_index = 0

        reader = csv.reader(data_lines[header_index:])
        header = next(reader)
        # Drop the prefix cell itself ("data") from the header and every row.
        prefix_cells = prefix.count(",")
        names = header[prefix_cells:]
        if not names:
            raise ScopikError(f"empty header after prefix in {path}")

        rows: list[list[str]] = []
        for row in reader:
            cells = row[prefix_cells:]
            if len(cells) != len(names):
                continue  # torn/partial line (serial capture can truncate mid-row)
            rows.append(cells)
        if not rows:
            raise ScopikError(f"no data rows in {path}")

        from scopik.sources import split_text_columns

        return split_text_columns(names, rows)
