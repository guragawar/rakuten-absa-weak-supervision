# -*- coding: utf-8 -*-
"""Shared utilities for parsing Rakuten Travel review panel TSV files (2009–2024).

Each yearly file has 18 tab-separated columns (no header). Some rows contain
embedded tabs inside the review text; the parser merges extras into column 17
and keeps column 18 as the hotel reply.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

N_FIELDS = 18

FIELD_NAMES = (
    "user_id",
    "posted_at",
    "hotel_id",
    "plan_id",
    "plan_name",
    "room_code",
    "room_name",
    "purpose",
    "companion",
    "overall_rating",
    "rating_1",
    "rating_2",
    "rating_3",
    "rating_4",
    "rating_5",
    "rating_6",
    "review_text",
    "reply_text",
)

PANEL_FILE_PATTERN = re.compile(r"^01_Travel_Review\.(\d{4})\.tsv$")


@dataclass(frozen=True)
class PanelRow:
    user_id: str
    posted_at: str
    hotel_id: str
    plan_id: str
    plan_name: str
    room_code: str
    room_name: str
    purpose: str
    companion: str
    overall_rating: str
    aspect_ratings: tuple[str, str, str, str, str, str]
    review_text: str
    reply_text: str
    source_year: int
    source_file: str
    line_no: int

    @property
    def review_text_norm(self) -> str:
        return normalize_review_text(self.review_text)

    @property
    def text_hash(self) -> str:
        return text_hash(self.review_text_norm)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_raw_dir() -> Path:
    return project_root() / "gold/data/raw"


def normalize_review_text(text: str) -> str:
    """Normalize Japanese review text for overlap / dedup matching."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop all whitespace so sentence-split differences do not break matching.
    text = re.sub(r"\s+", "", text)
    return text.strip()


def text_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16]


def discover_panel_files(raw_dir: Path | None = None) -> list[tuple[int, Path]]:
    """Return (year, path) pairs sorted by year for available panel files."""
    raw_dir = raw_dir or default_raw_dir()
    if not raw_dir.is_dir():
        return []

    found: list[tuple[int, Path]] = []
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        match = PANEL_FILE_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: item[0])


def fix_panel_fields(fields: list[str]) -> list[str] | None:
    """Repair a split TSV line to exactly N_FIELDS columns."""
    if len(fields) < N_FIELDS:
        return None
    if len(fields) == N_FIELDS:
        return fields
    # Embedded tabs: keep cols 1-16, merge middle into review_text, last = reply.
    return fields[:16] + ["\t".join(fields[16:-1]), fields[-1]]


def parse_panel_fields(
    fields: list[str],
    *,
    source_year: int,
    source_file: str,
    line_no: int,
) -> PanelRow | None:
    fixed = fix_panel_fields(fields)
    if fixed is None:
        return None
    return PanelRow(
        user_id=fixed[0].strip(),
        posted_at=fixed[1].strip(),
        hotel_id=fixed[2].strip(),
        plan_id=fixed[3].strip(),
        plan_name=fixed[4].strip(),
        room_code=fixed[5].strip(),
        room_name=fixed[6].strip(),
        purpose=fixed[7].strip(),
        companion=fixed[8].strip(),
        overall_rating=fixed[9].strip(),
        aspect_ratings=tuple(fixed[10 + i].strip() for i in range(6)),
        review_text=fixed[16].strip(),
        reply_text=fixed[17].strip(),
        source_year=source_year,
        source_file=source_file,
        line_no=line_no,
    )


def iter_panel_rows(
    path: Path,
    *,
    source_year: int | None = None,
) -> Iterator[PanelRow]:
    """Stream parsed rows from one yearly panel file."""
    if source_year is None:
        match = PANEL_FILE_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"Cannot infer year from filename: {path.name}")
        source_year = int(match.group(1))

    source_file = str(path)
    # errors="replace": tolerate the truncated 2016 dump (see parse_travel_review_panel.py)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            row = parse_panel_fields(
                fields,
                source_year=source_year,
                source_file=source_file,
                line_no=line_no,
            )
            if row is not None:
                yield row


def char_shingles(text: str, k: int = 3) -> set[str]:
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def minhash_signature(shingles: set[str], n_perm: int = 64) -> list[int]:
    if not shingles:
        return [2**32 - 1] * n_perm
    sig: list[int] = []
    for seed in range(n_perm):
        min_hash = 2**32
        for shingle in shingles:
            digest = hashlib.md5(f"{seed}:{shingle}".encode("utf-8")).hexdigest()
            value = int(digest[:8], 16)
            if value < min_hash:
                min_hash = value
        sig.append(min_hash)
    return sig


def minhash_similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)
