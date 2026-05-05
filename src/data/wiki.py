"""FEVER wiki title encode/decode helpers.

FEVER's Wikipedia dump uses a custom title encoding (`Inception_-LRB-2010_film-RRB-`).
HoVer stores titles in the same encoded form. We use these helpers when matching
gold titles against the dump and when rendering for human inspection.
"""

from __future__ import annotations

# FEVER wiki encoding: titles in the dump use this form.
_ENCODE_MAP = {
    " ": "_",
    "(": "-LRB-",
    ")": "-RRB-",
    "[": "-LSB-",
    "]": "-RSB-",
    ":": "-COLON-",
}


def fever_encode(title: str) -> str:
    """Encode a human-readable title to FEVER wiki-pages form."""
    out = title
    for k, v in _ENCODE_MAP.items():
        out = out.replace(k, v)
    return out


def fever_decode(title: str) -> str:
    """Decode a FEVER wiki-pages title back to human-readable form."""
    out = title
    for k, v in _ENCODE_MAP.items():
        out = out.replace(v, k)
    return out
