"""LLM connection point for the data layer (intended model: Qwen3.5-9B, shared with the query side).

Principle: rules run first; the LLM only receives the part the rules do NOT recognise, as a small summary
(never a whole table). LLM output must pass checks before use. With no model connected,
every function returns empty and the pipeline still runs as if only rules existed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.request

# OpenAI-compatible server (vLLM). In the contest: vLLM in Docker, one server per GPU.
# DATAMATE_LLM_URL unset -> available() = False, the pipeline runs rules only. Several servers: URLs separated by commas.
_URLS = [u.strip().rstrip("/") for u in os.environ.get("DATAMATE_LLM_URL", "").split(",") if u.strip()]
_URL = _URLS[0] if _URLS else ""
_turn = itertools.count()
_KEYS = [k.strip() for k in os.environ.get("DATAMATE_LLM_KEY", "").split(",") if k.strip()]
_KEY = _KEYS[0] if _KEYS else ""
_MODEL = os.environ.get("DATAMATE_LLM_MODEL", "")
# DATAMATE_LLM_EXTRA='{"reasoning_effort": "none"}': a field added to every request. Recent ollama turns on Qwen3.5's thinking mode by itself
# (ignoring the template) and it can only be turned off with this field; vLLM turns it off through chat_template_kwargs, so it is not needed there.
_EXTRA = json.loads(os.environ.get("DATAMATE_LLM_EXTRA") or "{}")
# DATAMATE_LLM_CACHE=<file.jsonl>: store answers by (model, prompt) -> reruns do not ask again, reproducible
_CACHE_PATH = os.environ.get("DATAMATE_LLM_CACHE", "")
_HAN_CHOT = float(os.environ.get("DATAMATE_HAN_CHOT") or 0)
_cache: dict | None = None
_lock = threading.Lock()


def _cache_key(body: dict) -> str:
    return hashlib.sha1(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _cache_load() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        if _CACHE_PATH and os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    _cache[r["k"]] = r["v"]
    return _cache


def _post(path: str, body: dict | None, timeout: float, tries: int = 3) -> dict:
    """Send to the servers in turn; on a network error or busy server retry (another server if any), report an error only when attempts run out."""
    for k in range(tries):
        if _HAN_CHOT:       # run.sh sets DATAMATE_HAN_CHOT: near the deadline the wait time is cut, past the deadline no more calls
            con = _HAN_CHOT - time.time()
            if con <= 5:
                raise TimeoutError("hết giờ thi (DATAMATE_HAN_CHOT)")
            timeout = min(timeout, con)
        idx = next(_turn) % len(_URLS) if _URLS else 0
        url = _URLS[idx] if _URLS else _URL
        key = _KEYS[idx] if idx < len(_KEYS) else _KEY      # one key per URL (DATAMATE_LLM_KEY separated by commas)
        req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if k == tries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code < 500):
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("không gọi được LLM")


def available() -> bool:
    return bool(_URL)


def model_name() -> str:
    global _MODEL
    if not _MODEL:
        _MODEL = _post("/v1/models", None, 30)["data"][0]["id"]
    return _MODEL


def _chat_raw(system: str, user: str, max_tokens: int = 512, timeout: float = 120, temperature: float = 0,
              n: int = 1, top_p: float | None = None, seed: int = 2026) -> list[str]:
    """Call /v1/chat/completions; n >= 1 takes n samples in one request (vLLM shares the prompt part)."""
    body = {"model": model_name(), "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}}
    if temperature:             # fixed so reruns give the same result (contest rule: two runs differ by at most 0.03)
        body["seed"] = seed
    body.update(_EXTRA)
    if n > 1:
        body["n"] = n
    if top_p is not None:       # after _EXTRA: a separate top_p (e.g. 0.9 when taking many samples) is not overridden by the shared top_p
        body["top_p"] = top_p
    key = _cache_key(body) if _CACHE_PATH else None
    if key:
        with _lock:
            hit = _cache_load().get(key)
        if hit is not None:
            return hit if isinstance(hit, list) else [hit]
    texts = [c["message"]["content"] for c in _post("/v1/chat/completions", body, timeout)["choices"]]
    if key:
        val = texts[0] if n == 1 else texts
        with _lock:
            _cache_load()[key] = val
            with open(_CACHE_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": key, "v": val}, ensure_ascii=False) + "\n")
    return texts


def chat_n(system: str, user: str, max_tokens: int = 512, timeout: float = 120, temperature: float = 0.7,
           n: int = 4, top_p: float | None = 0.9, seed: int = 2026) -> list[str]:
    """n samples in one request (4 SQL candidates at temperature 0.7)."""
    return _chat_raw(system, user, max_tokens=max_tokens, timeout=timeout, temperature=temperature, n=n, top_p=top_p,
                     seed=seed)


def chat(system: str, user: str, max_tokens: int = 512, timeout: float = 120, temperature: float = 0) -> str:
    """One query, temperature 0 by default, thinking mode off (Qwen3.5 has it on by default -> slow, long)."""
    return _chat_raw(system, user, max_tokens=max_tokens, timeout=timeout, temperature=temperature, n=1)[0]
