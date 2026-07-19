"""Plain header CSV, e.g. finn's logs/lqr_sim/<ts>/timeseries.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scopik.datamodel import ScopikError
from scopik.profile import Profile


class CsvSource:
    def read_table(
        self, path: Path, profile: Profile
    ) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                names = next(reader)
            except StopIteration:
                raise ScopikError(f"empty CSV: {path}") from None
            rows = [row for row in reader if len(row) == len(names)]
        if not rows:
            raise ScopikError(f"no data rows in {path}")

        from scopik.sources import split_text_columns

        return split_text_columns(names, rows)
