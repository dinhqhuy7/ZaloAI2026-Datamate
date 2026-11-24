"""Normalise questions in bulk via the qwen_norm model (mask <ENT_X> from DuckDB -> LLM -> Guardrail -> Unmask).

Supports standalone runs on every dataset (a contest set, examples_train, examples_luat)
using only the shop's DuckDB, with a per-question cache so a resume is instant.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import duckdb

from datamate.loader.parsers import fold
from datamate.query.compose import asks_yes_no, near_rules

NORM_SYSTEM_PROMPT = (
    "Bạn là chuyên gia chuẩn hóa câu hỏi tiếng Việt cho bài toán Text-to-SQL quản lý bán hàng.\n"
    "Nhiệm vụ: Sửa lỗi chính tả, teencode, viết tắt, thêm dấu và chuẩn hóa câu hỏi rõ ràng về hình thức và kiểu trả về (bảng, danh sách, %).\n"
    "Nếu có mục \"Luật liên quan:\", PHẢI giữ nguyên đúng tên thuật ngữ trong luật, không tự ý đổi sang chỉ số khác và không tự chèn giải thích dài trong ngoặc.\n"
    "TUYỆT ĐỐI GIỮ NGUYÊN các thẻ placeholder <ENT_X>. Chỉ trả về đúng 1 câu chuẩn hóa."
)

# Regexes that catch syntactic entities (strings in brackets, dates / months, SKU / order codes, money with dot separators)
_QUOTE_RE = re.compile(r'("[^"]+"|\'[^\']+\')')
# Date DD/MM/YYYY or month M/YYYY, M/YY, M-YYYY (does not catch the number after the letter Q / "quý")
_DATE_RE = re.compile(
    r"(?<![\d/.-])(?:\d{1,2}[/-]\d{1,2}[/-](?:20\d\d|\d\d)|(?:1[0-2]|0?[1-9])[/-](?:20\d\d|[23]\d))(?![\d/.-])"
)
_SKU_RE = re.compile(r"\b(?:[A-Za-z]{1,5}[-_]\d+[A-Za-z0-9]*|(?!Q[1-4]\b|T\d{1,2}\b|L\d+\b)[A-Za-z]{2,5}\d{2,}[A-Za-z0-9]*)\b")
_MONEY_DOT_RE = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")

# Common words are not treated as entity names when too short
_GENERIC_ENT_STOP = {
    "tra", "banh", "keo", "sua", "nuoc", "thuoc", "hang", "don", "khach", "khach le", "khach vang lai",
    " tiền mặt", "tien mat", "chuyen khoan", "tai quay", "dien thoai", "zalo", "shopee", "grab",
    "hoan thanh", "da giao", "huy", "tra hang", "dang giao", "cod", "ghi no", "vi", "the",
}

_shop_ent_cache: dict[str, list[str]] = {}
_shop_rule_cache: dict[str, str] = {}
_shop_ent_lock = threading.Lock()
_FOREIGN_RE = re.compile(r"[\u0400-\u04FF\u4E00-\u9FFF]")
_PAREN_RE = re.compile(r"\(([^()]{2,80})\)")


def load_shop_rules(db_path: str) -> str:
    """Read the whole _rulebook from the shop's DuckDB."""
    with _shop_ent_lock:
        if db_path in _shop_rule_cache:
            return _shop_rule_cache[db_path]
    if not db_path or not os.path.exists(db_path):
        return ""
    txt = ""
    try:
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute("SELECT text FROM _rulebook ORDER BY seq").fetchall()
        con.close()
        txt = "\n".join(r[0] or "" for r in rows)
    except Exception:
        pass
    with _shop_ent_lock:
        _shop_rule_cache[db_path] = txt
    return txt


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def load_shop_entities(db_path: str) -> list[str]:
    """Read the whole catalogue of product names, product groups, customer groups, active ingredients, customers, staff, suppliers from the shop's DuckDB."""
    with _shop_ent_lock:
        if db_path in _shop_ent_cache:
            return _shop_ent_cache[db_path]
    if not db_path or not os.path.exists(db_path):
        return []
    ents: set[str] = set()
    try:
        con = duckdb.connect(db_path, read_only=True)
        tables = con.execute("SELECT table_name, role FROM _meta_tables").fetchall()
        target_canons = {
            "product_name", "category", "active_ingredient", "brand",
            "customer_name", "customer_group", "supplier_name", "staff_name",
            "driver_name", "expense_category", "vehicle", "service_desc",
            "pet_name", "branch", "warehouse",
        }
        _plate_suffix_re = re.compile(r"\s+\d{2}[A-Za-z]\d[-.\s]\d+.*$")
        for t, _ in tables:
            cols = con.execute(
                "SELECT column_name, canonical FROM _meta_columns WHERE table_name = ?", [t]
            ).fetchall()
            for col, canon in cols:
                if canon in target_canons:
                    try:
                        vals = con.execute(
                            f'SELECT DISTINCT CAST("{col}" AS VARCHAR) FROM "{t}" WHERE "{col}" IS NOT NULL LIMIT 3000'
                        ).fetchall()
                        for (v,) in vals:
                            v = str(v).strip()
                            if len(v) >= 4 and fold(v) not in _GENERIC_ENT_STOP and not v.isdigit():
                                ents.add(v)
                            if canon == "vehicle":
                                prefix = _plate_suffix_re.sub("", v).strip()
                                if len(prefix) >= 4 and fold(prefix) not in _GENERIC_ENT_STOP:
                                    ents.add(prefix)
                    except Exception:
                        pass
        # Add _product_aliases if present
        try:
            for a, p in con.execute("SELECT alias, product_name FROM _product_aliases").fetchall():
                for v in (a, p):
                    if v and len(str(v).strip()) >= 4:
                        ents.add(str(v).strip())
        except Exception:
            pass
        con.close()
    except Exception:
        pass
    out = sorted(ents, key=len, reverse=True)
    with _shop_ent_lock:
        _shop_ent_cache[db_path] = out
    return out


def extract_and_mask(raw_q: str, shop_ents: list[str], known_ents: list[str] | None = None) -> tuple[str, dict[str, str]]:
    """Replace the entities in the question with <ENT_0>, <ENT_1>... in order of decreasing length."""
    candidates: list[str] = []
    if known_ents:
        candidates.extend([e for e in known_ents if e and e in raw_q])
    else:
        # 1. Strings in brackets
        for m in _QUOTE_RE.finditer(raw_q):
            candidates.append(m.group(0))
        # 2. Entities from the shop's DB (matched both with and without accents at word boundaries)
        raw_low = raw_q.lower()
        raw_strip = _strip_accents(raw_low)
        for ent in shop_ents:
            ent_strip = _strip_accents(ent.lower())
            if len(ent_strip) < 4:
                continue
            idx = raw_strip.find(ent_strip)
            if idx != -1:
                end = idx + len(ent_strip)
                before_ok = (idx == 0 or not raw_strip[idx - 1].isalnum())
                after_ok = (end == len(raw_strip) or not raw_strip[end].isalnum())
                if before_ok and after_ok and len(raw_q[idx:end]) == len(ent):
                    candidates.append(raw_q[idx:end])
        # 3. Dates / month-year
        for m in _DATE_RE.finditer(raw_q):
            # Do not cut if it is preceded by the word "quý" / "q"
            prefix = fold(raw_q[:m.start()])
            if not re.search(r"\b(?:quy|q)\s*$", prefix):
                candidates.append(m.group(0))
        # 4. SKU / staff code / customer code
        for m in _SKU_RE.finditer(raw_q):
            candidates.append(m.group(0))
        # 5. Money with dot separators
        for m in _MONEY_DOT_RE.finditer(raw_q):
            candidates.append(m.group(0))

    # Deduplicate and sort by decreasing length (matches the standard of full_gold_norm_chatml.jsonl)
    unique_ents = []
    seen = set()
    for c in sorted(candidates, key=len, reverse=True):
        if c and c not in seen:
            seen.add(c)
            unique_ents.append(c)

    masked = raw_q
    ent_map: dict[str, str] = {}
    ent_idx = 0
    for ent in unique_ents:
        if ent in masked:
            # Do not replace inside an already created <ENT_...> tag
            parts = re.split(r"(<ENT_\d+>)", masked)
            replaced = False
            for k in range(0, len(parts), 2):
                if ent in parts[k]:
                    tag = f"<ENT_{ent_idx}>"
                    parts[k] = parts[k].replace(ent, tag)
                    ent_map[tag] = ent
                    replaced = True
            if replaced:
                masked = "".join(parts)
                ent_idx += 1
    return masked, ent_map


# Regex that cuts a redundant "là bao nhiêu " before "dưới dạng danh sách" in name / listing questions
_REDUNDANT_BAO_NHIEU_LIST = re.compile(
    r"((?:là\s+những\s+[^?]+|là\s+gì|là\s+ai|những\s+[^?]+\s+nào|liệt\s+kê\s+[^?]+?))\s+là\s+bao\s+nhiêu\s+(dưới\s+dạng\s+danh\s+sách)",
    re.IGNORECASE,
)
# A question starting with "Cho tôi <N> mặt hàng/khách..." or "Liệt kê <N>..." that ends with "là bao nhiêu dưới dạng danh sách"
_IMPERATIVE_LIST_BAO_NHIEU = re.compile(
    r"^((?:cho\s+tôi|cho\s+mình|giúp\s+mình\s+liệt\s+kê|liệt\s+kê)\s+\d+\s+(?:mặt\s+hàng|món|sản\s+phẩm|hàng|khách|khách\s+hàng)[^?]*?)\s+là\s+bao\s+nhiêu\s+(dưới\s+dạng\s+danh\s+sách)",
    re.IGNORECASE,
)


# Abbreviated money: "3tr1" -> "3,1 triệu", "3tr" -> "3 triệu", "54k" -> "54.000 đồng". Runs on a question whose entities are ALREADY masked.
_PRE_MONEY_TR_DEC = re.compile(r"(?<![\d.,])(\d+)\s*tr(\d)(?!\d)\b", re.IGNORECASE)
_PRE_MONEY_TR = re.compile(r"(?<![\d.,])(\d+)\s*tr\b", re.IGNORECASE)
_PRE_MONEY_K = re.compile(r"(?<![\d.,])(\d+)\s*k\b", re.IGNORECASE)
_FAKE_TABLE_RE = re.compile(r"\s*,?\s*dưới\s+dạng\s+bảng\b", re.IGNORECASE)


def _pre_norm_money(text: str) -> str:
    t = _PRE_MONEY_TR_DEC.sub(r"\1,\2 triệu", text)
    t = _PRE_MONEY_TR.sub(r"\1 triệu", t)
    return _PRE_MONEY_K.sub(r"\1.000 đồng", t)


def _unmask(text: str, ent_map: dict[str, str]) -> str:
    for tag, orig in sorted(ent_map.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(tag, orig)
    return text


def postprocess_and_validate(raw_q: str, pred_masked: str, ent_map: dict[str, str]) -> tuple[str, bool]:
    """Run the 3-layer Guardrail check and Unmask to return a safely normalised question."""
    if not pred_masked or not pred_masked.strip():
        return raw_q, False

    pred = pred_masked.strip().strip('"').strip("'")
    if "Câu chuẩn hóa:" in pred:
        pred = pred.split("Câu chuẩn hóa:")[-1].strip()

    # Layer 1: check that 100% of <ENT_k> tags are preserved and no foreign-language text slips in
    if _FOREIGN_RE.search(pred):
        return raw_q, False
    expected_tags = set(ent_map.keys())
    found_tags = set(re.findall(r"<ENT_\d+>", pred))
    if expected_tags != found_tags:
        return raw_q, False

    # Block the 4B error of adding "Tháng " before <ENT_k> when <ENT_k> is a person / entity name rather than a month / year
    _month_ent_re = re.compile(r"^(?:0?[1-9]|1[0-2])[/-](?:20)?\d{2}$")
    m_thang = re.match(r"^(Tháng)\s+(<ENT_\d+>)(?!\w)", pred)
    if m_thang:
        tag = m_thang.group(2)
        orig_val = ent_map.get(tag, "")
        if not _month_ent_re.match(orig_val) and not re.search(rf"\b(?:tháng|th|t)\s*{re.escape(orig_val)}", raw_q, re.I):
            pred = pred[len(m_thang.group(1)):].lstrip()

    unmasked = pred
    for tag, orig in sorted(ent_map.items(), key=lambda x: len(x[0]), reverse=True):
        unmasked = unmasked.replace(tag, orig)

    # Remove long explanatory brackets inserted by itself (keep only (%) and brackets already in raw_q)
    def _clean_paren(m: re.Match) -> str:
        inner = m.group(1).strip()
        if inner == "%" or m.group(0) in raw_q:
            return m.group(0)
        return ""

    unmasked = _PAREN_RE.sub(_clean_paren, unmasked)

    # Clean up extra spaces around the month / year slash if any
    unmasked = re.sub(r"\s+", " ", unmasked).strip()

    # Layer 2: preserve the nature of Yes/No (Boolean) questions, Quarter (Q1..Q4) and money questions
    if asks_yes_no(raw_q) and not asks_yes_no(unmasked):
        return raw_q, False
    m_quy = re.search(r"\bQ([1-4])\s*[/-]\s*(?:20)?\d{2}\b", raw_q, re.I)
    if m_quy and not re.search(rf"\b(?:qu[yý]\s*{m_quy.group(1)}|Q{m_quy.group(1)})\b", unmasked, re.I):
        return raw_q, False
    if re.search(r"(?:bao\s*nhi[eê]u|bn|nhi[eê]u)\s*ti[eề]n", raw_q, re.I):
        if re.search(r"\bbao\s+nhiêu\s+(?:khách|đơn|mặt\s+hàng|sản\s+phẩm|món|nhân\s+viên|thợ)\b", unmasked, re.I) and not re.search(r"tiền|doanh\s+thu|giá\s+trị|tổng\s+nợ|công\s+nợ", unmasked, re.I):
            return raw_q, False

    # Layer 3: clean the redundant "là bao nhiêu " before "dưới dạng danh sách" in questions that ask for object names
    unmasked = _REDUNDANT_BAO_NHIEU_LIST.sub(r"\1 \2", unmasked)
    unmasked = _IMPERATIVE_LIST_BAO_NHIEU.sub(r"\1 \2", unmasked)

    # Layer 4: clean repeated words, a stray "nhiêu" and add "Tháng " before a sentence-initial M/YY
    unmasked = re.sub(r"\b([Gg]iá trị)\s+giá trị\b", r"\1", unmasked)
    unmasked = re.sub(r"\b(đơn hàng)\s+hàng\b", r"\1", unmasked, flags=re.I)
    unmasked = re.sub(r"\b([Dd]oanh thu)\s+doanh thu\b", r"\1", unmasked)
    unmasked = re.sub(r"\b([Tt]ổng)\s+tổng\b", r"\1", unmasked)
    unmasked = re.sub(r"\bbnhiu\b", "bao nhiêu", unmasked, flags=re.I)
    unmasked = re.sub(r"(?<!bao\s)(?<!bao)\bnhiêu\b", "bao nhiêu", unmasked, flags=re.I)
    unmasked = re.sub(r"^((?:0?[1-9]|1[0-2])[/-](?:25|26|2025|2026))\b", r"Tháng \1", unmasked)
    unmasked = re.sub(r"\b(trong|của|ở|vào|so sánh|xét)\s+((?:0?[1-9]|1[0-2])[/-](?:25|26|2025|2026))\b", r"\1 tháng \2", unmasked, flags=re.I)
    unmasked = re.sub(r"\s+", " ", unmasked).strip()

    # Layer 4b: 4B adds "dưới dạng bảng" when the original question does not mention a table (§9: a table only when the question asks for one)
    if "bảng" not in raw_q.lower() and "bang" not in raw_q.lower():
        unmasked = _FAKE_TABLE_RE.sub("", unmasked)
        unmasked = re.sub(r"\s+([?!.])", r"\1", unmasked)

    # Layer 4c: 4B drops the number's separator ("3,1 triệu" -> "31 triệu"): fall back to the input question
    raw_nums = re.findall(r"\d+(?:[.,]\d+)?", raw_q)
    pred_nums = re.findall(r"\d+(?:[.,]\d+)?", unmasked)
    if len(raw_nums) == len(pred_nums):
        for rn, pn in zip(raw_nums, pred_nums):
            if ("," in rn or "." in rn) and rn != pn and "," not in pn and "." not in pn \
                    and re.sub(r"[.,]", "", rn) == pn:
                return raw_q, False

    return unmasked, True


def call_norm_llm(url: str, key: str, model: str, masked_q: str, tries: int = 3) -> str:
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 180,
        "messages": [
            {"role": "system", "content": NORM_SYSTEM_PROMPT},
            {"role": "user", "content": masked_q},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": "Mozilla/5.0",
    }
    for k in range(tries):
        try:
            req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as r:
                resp = json.loads(r.read().decode("utf-8"))
                return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if k == tries - 1:
                raise RuntimeError(f"Lỗi gọi qwen_norm: {e}") from e
            time.sleep(1.5 * (k + 1))
    return masked_q


def normalize_records(
    records: list[dict],
    db_root: str,
    url: str,
    key: str,
    model: str,
    cache_path: str,
    known_entities_map: dict[str, list[str]] | None = None,
    workers: int = 24,
) -> list[dict]:
    """Normalise a list of question records with a persistent cache (a rerun costs 0s for questions already done)."""
    cache: dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    cache[r["k"]] = r["v"]

    cache_lock = threading.Lock()
    t0 = time.time()
    done_cnt = 0
    guardrail_revert_cnt = 0

    loi: list = []      # the call to 4B failed (keep the original question)

    def process_one(rec: dict) -> dict:
        nonlocal done_cnt, guardrail_revert_cnt
        t_q0 = time.time()
        qid = rec.get("id", "")
        shop = rec.get("business_id") or rec.get("shop") or ""
        raw_q = rec.get("goc") or rec["question"]
        db_path = os.path.join(db_root, shop, f"{shop}.duckdb") if db_root and shop else ""
        shop_ents = load_shop_entities(db_path) if db_path else []
        shop_rules = load_shop_rules(db_path) if db_path else ""
        known_ents = known_entities_map.get(qid) if known_entities_map else None

        masked_q, ent_map = extract_and_mask(raw_q, shop_ents, known_ents)
        masked_q = _pre_norm_money(masked_q)        # after masking: product names inside <ENT_x> are not changed
        prep_q = _unmask(masked_q, ent_map)         # safe input question (used when 4B is broken / blocked)
        near = near_rules(shop_rules, raw_q, k=3) if shop_rules else []
        if near:
            user_content = "Luật liên quan:\n" + "\n".join(x.strip() for x in near if x.strip()) + f"\n\nCâu hỏi: {masked_q}"
        else:
            user_content = f"Câu hỏi: {masked_q}"
        ckey = f"v2||{model}||{user_content}"

        with cache_lock:
            pred_masked = cache.get(ckey)

        if pred_masked is None:
            try:
                pred_masked = call_norm_llm(url, key, model, user_content)
            except RuntimeError as e:       # 4B failed on ONE question (timeout, too long): that question keeps its original, the other questions are still cleaned
                print(f"  [Norm] câu {qid} gọi 4B lỗi, giữ câu gốc: {str(e)[:120]}", flush=True)
                loi.append(qid)
                return {**rec, "question": raw_q, "goc": raw_q}
            with cache_lock:
                cache[ckey] = pred_masked
                if cache_path:
                    with open(cache_path, "a", encoding="utf-8") as cf:
                        cf.write(json.dumps({"k": ckey, "v": pred_masked}, ensure_ascii=False) + "\n")

        norm_q, ok = postprocess_and_validate(prep_q, pred_masked, ent_map)
        with cache_lock:
            done_cnt += 1
            if not ok:
                guardrail_revert_cnt += 1
            if done_cnt % 100 == 0 or done_cnt == len(records):
                elapsed = time.time() - t0
                print(
                    f"  [Norm] {done_cnt}/{len(records)} câu ({elapsed:.1f}s, "
                    f"guardrail_fallback={guardrail_revert_cnt})",
                    flush=True,
                )

        out = dict(rec)
        out["goc"] = raw_q
        out["question"] = norm_q
        out["_norm_ms"] = max(0, int((time.time() - t_q0) * 1000))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(process_one, records))
    if len(loi) > 0.2 * len(records):   # 4B is completely broken: report an error so run.sh uses the original questions + the original example store + a Telex fix for the whole contest set
        raise RuntimeError(f"{len(loi)}/{len(records)} câu gọi 4B lỗi")
    return results


def main():
    ap = argparse.ArgumentParser(description="Sinh bộ câu hỏi chuẩn hóa qua model qwen_norm")
    ap.add_argument("--input", required=True, help="Input JSONL file (questions.jsonl or examples_*.jsonl)")
    ap.add_argument("--output", required=True, help="Output JSONL file after normalisation")
    ap.add_argument("--db-root", default="", help="Folder holding the shops' DuckDB files (e.g. work/db/<contest set>)")
    ap.add_argument("--url", default=os.environ.get("DATAMATE_NORM_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--key", default=os.environ.get("DATAMATE_NORM_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("DATAMATE_NORM_MODEL", "qwen_norm"))
    ap.add_argument("--cache", default="data/norm_cache_qwen_norm_4b_v2.jsonl", help="File caching the results of calls to qwen_norm")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    if a.cache:
        os.makedirs(os.path.dirname(os.path.abspath(a.cache)), exist_ok=True)

    known_entities_map: dict[str, list[str]] = {}

    with open(a.input, encoding="utf-8-sig") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(
        f">>> Bắt đầu Normalize {len(records)} câu từ {a.input} -> {a.output} "
        f"(workers={a.workers}, model={a.model})",
        flush=True,
    )
    norm_recs = normalize_records(
        records=records,
        db_root=a.db_root,
        url=a.url,
        key=a.key,
        model=a.model,
        cache_path=a.cache,
        known_entities_map=known_entities_map or None,
        workers=a.workers,
    )

    with open(a.output, "w", encoding="utf-8") as out_f:
        for r in norm_recs:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    changed = sum(1 for r in norm_recs if r["question"] != r["goc"])
    print(
        f">>> HOÀN TẤT: {a.output} | Đã chuẩn hóa thay đổi: {changed}/{len(norm_recs)} câu ({changed*100.0/max(len(norm_recs),1):.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
