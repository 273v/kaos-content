"""Shared text canonicalization for dedup, search, and extraction.

This is the **single source of truth** for how KAOS normalizes text
before hashing, comparing, or deduplicating. Every module that needs
a "canonical form" of a string — the dedup pipeline, the extraction
cell dedup, the alpha-LLM merger, the citation verifier's normalized
token strategy — MUST use the functions here.

Prior to this module, the codebase had three independent
implementations of entity-suffix stripping, two incompatible date
normalizers, and divergent whitespace/unicode handling. This module
consolidates them.

Functions:

- :func:`normalize_text` — whitespace collapse + lowercase + NFKC.
  The universal baseline all other normalizers compose on.
- :func:`normalize_entity` — text normalization + strip trailing
  corporate suffixes (108-entry gazetteer ported from
  AlphaEntityExtractor).
- :func:`normalize_date` — parse common date formats → ISO 8601.
  Falls back to text normalization on parse failure (never returns
  None — callers that hash the result need a deterministic string).
- :func:`normalize_number` — strip currency symbols + thousands
  separators → Decimal string (preserves precision: "13.50" stays
  "13.50", not "13.5").
- :func:`canonical_hash` — SHA-256 of any canonical string.
- :func:`clean_extracted_text` — noncharacter / soft-hyphen handling + NFKC
  for text extracted from PDFs, OCR and HTML (keeps case and whitespace).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from decimal import Decimal, InvalidOperation

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# 108-entry entity suffix gazetteer. Ported from
# kaos-llm-core/extract/alpha/entity.py's 108-suffix set. Lowercased
# and period-stripped for uniform matching.
ENTITY_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "llc",
        "ltd",
        "limited",
        "co",
        "company",
        "plc",
        "gmbh",
        "ag",
        "sa",
        "sarl",
        "bv",
        "nv",
        "pty",
        "lp",
        "llp",
        "se",
        "kg",
        "ohg",
        "ev",
        "kgaa",
        "ug",
        "mbh",
        "spa",
        "srl",
        "snc",
        "sas",
        "eurl",
        "sci",
        "gie",
        "sl",
        "sll",
        "sa de cv",
        "sab",
        "ab",
        "aps",
        "as",
        "is",
        "hf",
        "oy",
        "oyj",
        "ky",
        "kk",
        "gk",
        "nk",
        "yk",
        "brt",
        "cv",
        "vof",
        "maatschap",
        "ans",
        "da",
        "ks",
        "sp",
        "sro",
        "spol",
        "kft",
        "rt",
        "zrt",
        "bt",
        "pt",
        "tbk",
        "ud",
        "fa",
        "pte",
        "bhd",
        "sdn",
        "ltda",
        "sab de cv",
        "eireli",
        "mei",
        "holdings",
        "group",
        "partners",
        "associates",
        "trust",
        "fund",
        "ventures",
        "capital",
        "management",
        "services",
        "solutions",
        "technologies",
        "systems",
        "international",
        "global",
        "worldwide",
        "enterprises",
    }
)


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    unicode_nfkc: bool = True,
    strip_punctuation: bool = False,
) -> str:
    """Universal text normalization: whitespace collapse + optional transforms.

    This is the baseline that all other normalizers compose on. Callers
    that need entity/date/number normalization call the specialized
    functions, which call this internally.
    """
    result = text
    if unicode_nfkc:
        result = unicodedata.normalize("NFKC", result)
    if lowercase:
        result = result.lower()
    if strip_punctuation:
        result = _PUNCT_RE.sub("", result)
    return _WHITESPACE_RE.sub(" ", result).strip()


def is_noncharacter(ch: str) -> bool:
    """Return whether ``ch`` is a Unicode noncharacter code point.

    Noncharacters are U+FDD0..U+FDEF and the last two code points of every plane
    (U+xFFFE, U+xFFFF). They are reserved for internal use and are never valid in
    interchanged text (Unicode Standard, section 23.7).
    """
    o = ord(ch)
    return 0xFDD0 <= o <= 0xFDEF or (o & 0xFFFE) == 0xFFFE


SOFT_HYPHEN = "\u00ad"


def clean_extracted_text(text: str, *, unicode_nfkc: bool = True) -> str:
    """Normalize text extracted from PDF text layers, OCR or HTML for storage and training.

    Character-class rules only; nothing here matches words or rewrites prose:

    * **Noncharacters** (:func:`is_noncharacter`, e.g. U+FFFE, which older PDFium builds emit as
      an end-of-line hyphen marker) are treated like a SOFT HYPHEN when they end a line, and
      removed anywhere else.
    * **SOFT HYPHEN** (U+00AD) is default-ignorable: it marks an optional break and is never
      displayed except at a line break. At the end of a line whose next line starts with a
      lowercase letter, the two lines are joined into one word (the line break, surrounding
      spaces and the soft hyphen are dropped), the same rule kaos-pdf applies to PDFium's
      hyphen markers. Any other soft hyphen is removed.
    * **NFKC** (``unicode_nfkc``, default on), applied last.

    Line endings, other whitespace and all other characters are left as they are.
    """
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == SOFT_HYPHEN or is_noncharacter(ch):
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] in "\r\n":
                k = j + 1 if not (text[j] == "\r" and j + 1 < n and text[j + 1] == "\n") else j + 2
                m = k
                while m < n and text[m] in " \t":
                    m += 1
                prev_alpha = bool(out) and out[-1].isalpha()
                if prev_alpha and m < n and text[m].islower():
                    i = m  # join the word: drop marker, line break and indentation
                    continue
            i += 1  # drop the marker; keep whatever follows
            continue
        out.append(ch)
        i += 1
    result = "".join(out)
    return unicodedata.normalize("NFKC", result) if unicode_nfkc else result


def normalize_entity(text: str) -> str:
    """Text normalization + strip trailing corporate suffixes.

    Uses the 108-entry gazetteer (same set as AlphaEntityExtractor).
    Strips all trailing tokens that match a known suffix, so
    "Acme Corp." → "acme", "Beta Holdings LLC" → "beta".

    Handles:
    - Single-token suffixes: "Corp", "Corp.", "LLC"
    - Multi-word suffixes: "SA de CV", "SAB de CV"
    - Dotted tokens: "K.K." → strips dots then matches "kk"
    """
    normalized = normalize_text(text)
    tokens = normalized.split()
    changed = True
    while changed and tokens:
        changed = False
        # Try multi-word suffixes first (longest match)
        for n_words in range(min(3, len(tokens)), 0, -1):
            tail = " ".join(tokens[-n_words:])
            tail_clean = _strip_suffix_punctuation(tail).replace(".", "")
            if tail_clean in ENTITY_SUFFIXES:
                tokens = tokens[:-n_words]
                changed = True
                break
            # Also try with dots preserved (e.g., "s.a." → "sa" in the set)
            if n_words == 1:
                single = _strip_suffix_punctuation(tokens[-1]).replace(".", "")
                if single in ENTITY_SUFFIXES:
                    tokens.pop()
                    changed = True
                    break
    return " ".join(tokens) if tokens else normalized


def normalize_date(text: str) -> str:
    """Parse common date formats → ISO 8601 string.

    Falls back to :func:`normalize_text` on parse failure so the
    result is always a deterministic hashable string (never None).
    """
    import datetime as _dt

    stripped = text.strip()
    for fmt in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = _dt.datetime.strptime(stripped, fmt)
            return dt.date().isoformat() if "T" not in fmt and "H" not in fmt else dt.isoformat()
        except ValueError:
            continue
    # ISO fast path for dates that strptime missed
    try:
        return _dt.date.fromisoformat(stripped).isoformat()
    except ValueError:
        pass
    return normalize_text(stripped)


def normalize_number(text: str) -> str:
    """Strip currency symbols + thousands separators → Decimal string.

    Uses fixed-point notation so 1000000 stays "1000000" (not "1E+6")
    and trailing zeros are stripped for consistent hashing ("13.50" →
    "13.5"). The priority is DETERMINISTIC HASHING — the same numeric
    value must always produce the same string.
    """
    cleaned = text.strip()
    for symbol in ("$", "€", "£", "¥", "₹", "₩", "CHF", "USD", "EUR", "GBP"):
        cleaned = cleaned.replace(symbol, "")
    cleaned = cleaned.replace(",", "").strip()
    if not cleaned:
        return normalize_text(text)
    try:
        d = Decimal(cleaned)
        # Use fixed-point notation: Decimal("1E+6") → "1000000"
        # Normalize removes trailing zeros: "13.50" → "13.5"
        # This is intentional — two values are the same number iff
        # they have the same normalized Decimal form.
        return format(d.normalize(), "f")
    except InvalidOperation:
        return normalize_text(text)


def canonical_hash(canonical: str) -> str:
    """SHA-256 of a canonical string, hex-encoded."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strip_suffix_punctuation(token: str) -> str:
    """Strip trailing periods, commas, semicolons from a token for suffix matching."""
    return token.rstrip(".,;:")


__all__ = [
    "ENTITY_SUFFIXES",
    "canonical_hash",
    "normalize_date",
    "normalize_entity",
    "normalize_number",
    "normalize_text",
]
