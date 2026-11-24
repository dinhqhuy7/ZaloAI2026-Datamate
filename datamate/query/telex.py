"""Fix Telex typing errors in a question ("thangs" -> "tháng", "naof" -> "nào", "ddown" -> "đơn").

Accepted only when the result is a real Vietnamese syllable (am_tiet_viet.txt, drawn from the public word list Viet74K),
so proper names, product codes, English words are not changed. Unaccented Vietnamese has no syllable ending in s/f/r/x/j,
so an ordinary unaccented question stays as is.
Turn off: DATAMATE_TELEX=0.
"""
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ON = os.environ.get("DATAMATE_TELEX", "1") != "0"

_DAU = {"s": "́", "f": "̀", "r": "̉", "x": "̃", "j": "̣"}   # sắc huyền hỏi ngã nặng (the five Vietnamese tone marks)
_NGUYEN_AM = set("aăâeêioôơuưy")
_TIENG_ANH = {"bus", "gas", "plus", "yes", "kids", "ads", "items", "orders", "sales", "bills", "ships", "sells", "products",
              "shops", "has", "was", "is", "its", "this", "us", "tips", "tax", "max", "box", "mix", "fix", "six",
              "sex", "fax", "lux", "pax", "vs", "pcs", "kgs", "ms", "mr", "dr", "jr", "sr"}
# present in the Viet74K list (loanwords) but when a shop owner types a message it is almost always a typo
_LUON_GIAI = {"cos": "có", "as": "á"}
_BIEN = [("uwow", "ươ"), ("uow", "ươ"), ("uwo", "ươ"), ("aa", "â"), ("aw", "ă"), ("ee", "ê"), ("oo", "ô"),
         ("ow", "ơ"), ("uw", "ư"), ("dd", "đ")]


@lru_cache(maxsize=1)
def _am_tiet() -> frozenset:
    with open(Path(__file__).with_name("am_tiet_viet.txt"), encoding="utf-8") as fh:
        return frozenset(x.strip() for x in fh if x.strip() and not x.startswith("#"))


def _them_dau(goc: str, dau: str) -> str | None:
    at = _am_tiet()
    if not dau:
        return goc if goc in at else None
    # vowels with a hat / horn first, then from right to left ("quys" -> "quý")
    for i in sorted((i for i, ch in enumerate(goc) if ch in _NGUYEN_AM), key=lambda i: (goc[i] not in "ăâêôơư", -i)):
        ung = unicodedata.normalize("NFC", goc[:i + 1] + _DAU[dau] + goc[i + 1:])
        if ung in at:
            return ung
    return None


def giai_telex(tu: str) -> str:
    w = tu.lower()
    if w in _LUON_GIAI and not tu.isupper():
        kq = _LUON_GIAI[w]
        return kq.capitalize() if tu[0].isupper() else kq
    if len(w) < 2 or not w.isascii() or not w.isalpha() or w in _TIENG_ANH or w in _am_tiet():
        return tu
    if tu.isupper():                       # an all-uppercase word: abbreviation / code ("AA", "COD")
        return tu
    dau = w[-1] if w[-1] in _DAU else ""
    goc = w[:-1] if dau else w
    bien = any(a in goc for a, _ in _BIEN)
    if not dau and not bien:
        return tu
    if tu[0].isupper() and not bien:       # a proper name / English word with a capital initial: needs a clear Telex sign
        return tu
    if not dau and len(w) <= 2:            # "ee", "aa" standing alone: not certain enough
        return tu
    for a, b in _BIEN:
        goc = goc.replace(a, b)
    kq = _them_dau(goc, dau)
    if kq is None:
        return tu
    return kq.capitalize() if tu[0].isupper() else kq


def sua_cau(cau: str) -> str:
    if not ON:
        return cau
    t = unicodedata.normalize("NFC", str(cau))
    return re.sub(r"[A-Za-z]+", lambda m: giai_telex(m[0]), t)


def sua_q(q: dict) -> dict:
    """The question after the Telex fix (the original is kept in 'goc_telex' when something changed)."""
    t = sua_cau(q["question"])
    return q if t == q["question"] else {**q, "question": t, "goc_telex": q["question"]}
