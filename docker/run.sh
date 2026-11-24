#!/bin/bash
# Run inside Docker: build the DB from the data in /data, answer every question, write /result/submission.jsonl.
# One vLLM server per GPU; the code distributes questions across the servers. Each question is written as soon as it is done: if stopped midway,
# unwritten questions count as blank.
# Trial run outside Docker: set DATAMATE_LLM_URL (an existing server) to skip starting vLLM; APP, DATA, RESULT, WORK can be changed.
set -u
START=$(date +%s)
# maximum run time (default 19800 s = 5.5 h): after BUDGET seconds (counted from container start) questions not yet done are written blank, leaving time to write the files
BUDGET=${BUDGET:-19800}
APP=${APP:-/app}
DATA=${DATA:-/data}
RESULT=${RESULT:-/result}
WORK=${WORK:-/tmp/datamate}
MODEL=${MODEL:-/models/qwen}
# number of questions in parallel (each question queries the LLM several times, one after another): 2 GPUs x 64 sequences per vLLM server -> 32 threads
WORKERS=${WORKERS:-32}
# clean the questions with the 4B model (Qwen3.5-4B + merged LoRA v2) before answering; NORM=0 to turn it off
NORM=${NORM:-1}
NORM_MODEL=${NORM_MODEL:-/models/qwen_norm4b}
mkdir -p "$RESULT" "$RESULT/logs" "$WORK"
cd "$APP"
# fixed iteration order of Python string sets between runs (a rerun gives nearly the same results)
export PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0
# shared deadline: every LLM call (building the DB, answering) cuts its wait by the time left, and gives up past the deadline
export DATAMATE_HAN_CHOT=$((START + BUDGET))
# debt when the order table has no order total (NV2) + stock = purchases - shipments + returns (TON_DONG) + the data layer's self-repair loop (TU_SUA): on by default
export DATAMATE_NV2=${DATAMATE_NV2:-1} DATAMATE_TON_DONG=${DATAMATE_TON_DONG:-1} DATAMATE_TU_SUA=${DATAMATE_TU_SUA:-1}
# the Python with the libraries: the vLLM image only has python3 (the Dockerfile sets PY)
if [ -z "${PY:-}" ]; then
  for p in python3 python; do $p -c "import duckdb" 2>/dev/null && { PY=$p; break; }; done
fi
PY=${PY:-python3}

Q=$(find "$DATA" -name questions.jsonl | head -1)
if [ -z "$Q" ]; then echo "questions.jsonl not found in $DATA"; exit 1; fi
SPLIT=$(dirname "$Q")
# shop folder: data/ next to questions.jsonl as in train; if absent, the shops sit right next to questions.jsonl
SHOPS="$SPLIT/data"; [ -d "$SHOPS" ] || SHOPS="$SPLIT"
OUT="$RESULT/submission.jsonl"
LOG_OUT="$RESULT/logs/logs.jsonl"
touch "$OUT" "$LOG_OUT"

TU_BAT=0     # 1: the script starts the vLLM servers itself (inside Docker); 0: use existing servers (trial run)
if [ -z "${DATAMATE_LLM_URL:-}" ]; then
  TU_BAT=1
  NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
  # BATCH_INVARIANT=1: vLLM results do not depend on how requests are batched (VLLM_BATCH_INVARIANT, beta, slower) -> reruns match
  # more closely. Off by default (not yet measured on a real machine)
  BI=${BATCH_INVARIANT:-0}
  echo "question set: $SPLIT, GPU: $NGPU, Python: $PY, batch invariance: $BI"
  # Qwen3.5 has linear-attention layers (a Mamba-style cache): prefix caching for this kind is still experimental -> off by default
  # (PREFIX_CACHE=1 to turn on). Thinking mode is turned off per request (loader/llm.py sends enable_thinking=false).
  CACHE_FLAG="--no-enable-prefix-caching"; [ "${PREFIX_CACHE:-0}" = "1" ] && CACHE_FLAG="--enable-prefix-caching"
  # GPU_MEM: share of GPU memory for vLLM. 24 GB leaves no room for the KV cache at 0.80 (Qwen3.5-9B in bf16 takes about 17.7 GB): set 0.92
  GPU_MEM=${GPU_MEM:-0.80}
  FAST="--max-model-len 32768 --gpu-memory-utilization $GPU_MEM $CACHE_FLAG"
  # a server that dies at startup (out of memory, CUDA graph, odd flags, batch invariance...) is started once more with a less
  # demanding configuration, without batch invariance
  SAFE="--max-model-len 16384 --gpu-memory-utilization $GPU_MEM --enforce-eager"
  declare -a PIDS
  serve() {     # GPU, batch invariance 0/1, extra vLLM flags
    CUDA_VISIBLE_DEVICES=$1 VLLM_BATCH_INVARIANT=$2 vllm serve "$MODEL" --served-model-name qwen --port $((8000 + $1)) \
        --seed 2026 --max-num-seqs 32 "${@:3}" >> "$WORK/vllm_$1.log" 2>&1 &
    PIDS[$1]=$!
  }
  # stop the servers and WAIT until they have fully exited (releasing GPU memory): also clean up child processes (EngineCore) if needed
  stop_pids() {
    for p in "$@"; do pkill -TERM -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done
    for p in "$@"; do
      for i in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 2; done
      pkill -9 -P "$p" 2>/dev/null || true
      kill -9 "$p" 2>/dev/null || true
    done
    sleep 5
  }
  stop_all() { stop_pids "${PIDS[@]}"; }
  up() { $PY -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + '/v1/models', timeout=5)" "$1" 2>/dev/null; }
  wait_up() {   # URL, PID: succeeds when the server answers; fails when the process dies or after 20 minutes
    for i in $(seq 1 240); do
      up "$1" && return 0
      kill -0 "$2" 2>/dev/null || return 1
      sleep 5
    done
    return 1
  }
  GPUS=$(seq 0 $((NGPU > 0 ? NGPU - 1 : 0)))
  for g in $GPUS; do serve $g "$BI" $FAST; done
  LIVE=""
  for g in $GPUS; do
    u="http://127.0.0.1:$((8000 + g))"
    if ! wait_up "$u" "${PIDS[$g]}"; then
      echo "vLLM GPU $g did not come up, restarting with: $SAFE, without batch invariance"; tail -n 20 "$WORK/vllm_$g.log"
      kill "${PIDS[$g]}" 2>/dev/null; sleep 15
      serve $g 0 $SAFE
      # a server that still does not come up is dropped, continue with the remaining servers
      wait_up "$u" "${PIDS[$g]}" || { echo "vLLM GPU $g still did not come up:"; tail -n 30 "$WORK/vllm_$g.log"; continue; }
    fi
    LIVE="${LIVE:+$LIVE,}$u"
  done
  # restart 9B on every GPU (after the cleaning step): as the first time, with a fallback configuration
  start_9b() {
    LIVE=""
    for g in $GPUS; do serve $g "$BI" $FAST; done
    for g in $GPUS; do
      u="http://127.0.0.1:$((8000 + g))"
      if ! wait_up "$u" "${PIDS[$g]}"; then
        kill "${PIDS[$g]}" 2>/dev/null; sleep 15; serve $g 0 $SAFE
        wait_up "$u" "${PIDS[$g]}" || { echo "vLLM GPU $g still did not come up"; continue; }
      fi
      LIVE="${LIVE:+$LIVE,}$u"
    done
  }
  echo "vLLM ready after $(($(date +%s) - START))s"
  # sample by temperature only as in the measurements (ollama: top_p 1, top_k 0, repeat_penalty 1): do not let vLLM take the top_k / top_p
  # defaults from the model's generation_config.json
  export DATAMATE_LLM_URL="$LIVE" DATAMATE_LLM_KEY=none DATAMATE_LLM_MODEL=qwen \
         DATAMATE_LLM_EXTRA='{"top_p": 1, "top_k": -1, "min_p": 0, "repetition_penalty": 1}'
fi
echo "LLM: ${DATAMATE_LLM_URL:-none}"
if [ "$TU_BAT" = "1" ] && [ -z "${DATAMATE_LLM_URL:-}" ]; then
  echo "ERROR: no vLLM server could be started!" >&2
  exit 1
fi

# 1. data: read the shop files, assign canonical column names (LLM), build a DuckDB per shop. The column-name dictionary comes from the READMEs of other shops
# of the set; the extra READMEs (assets/readme_train, if present) only add names that the set's own READMEs never described (a set may have READMEs that do not describe columns)
LLM_FLAG=""; [ -n "$DATAMATE_LLM_URL" ] && LLM_FLAG="--llm"
BASE=""; [ -d assets/readme_train ] && BASE="--base-readmes assets/readme_train"
$PY -m datamate.run_all "$SHOPS" --out "$WORK/db" $LLM_FLAG --skip-tables $BASE
DB=$(find "$WORK/db" -mindepth 1 -maxdepth 1 -type d | head -1)
echo "DB built after $(($(date +%s) - START))s: $DB"

# 1b. clean the questions with 4B (<ENT_k> tags for names in the DB, the shop's relevant rules, safety checks; a question that does not pass
# the check keeps its original text). Any error falls back to the original question.
EX_TRAIN=assets/examples_train.jsonl; EX_LUAT=assets/examples_luat.jsonl; EX_SQL=assets/examples_sql_train.jsonl
QSPLIT="$SPLIT"; export DATAMATE_TELEX=1
if [ "$NORM" = "1" ] && [ -n "$DB" ] && [ -f assets/examples_train_norm.jsonl ]; then
  NURL=${DATAMATE_NORM_URL:-}
  if [ -z "$NURL" ] && [ "$TU_BAT" = "1" ] && [ -d "$NORM_MODEL" ]; then
    stop_all
    declare -a NPIDS
    for g in $GPUS; do
      CUDA_VISIBLE_DEVICES=$g vllm serve "$NORM_MODEL" --served-model-name qwen_norm --port $((8100 + g)) --seed 2026 \
          --max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.80 \
          --limit-mm-per-prompt '{"image": 0, "video": 0}' >> "$WORK/norm_$g.log" 2>&1 &
      NPIDS[$g]=$!
    done
    for g in $GPUS; do wait_up "http://127.0.0.1:$((8100 + g))" "${NPIDS[$g]}" && NURL="http://127.0.0.1:$((8100 + g))" && break; done
    echo "4B cleaning: ${NURL:-not up} after $(($(date +%s) - START))s"
  fi
  if [ -n "$NURL" ]; then
    mkdir -p "$WORK/de_norm"
    cp "$SPLIT"/rulebook.md "$SPLIT"/calendar.csv "$WORK/de_norm/" 2>/dev/null
    if $PY -m datamate.query.normalize_all --input "$Q" --output "$WORK/de_norm/questions.jsonl" --db-root "$DB" \
         --url "$NURL" --key "${DATAMATE_NORM_KEY:-none}" --model qwen_norm --cache "$WORK/norm_cache.jsonl" \
         --workers "${NORM_WORKERS:-32}" && [ -s "$WORK/de_norm/questions.jsonl" ]; then
      QSPLIT="$WORK/de_norm"; export DATAMATE_TELEX=0
      EX_TRAIN=assets/examples_train_norm.jsonl; EX_LUAT=assets/examples_luat_norm.jsonl; EX_SQL=assets/examples_sql_train_norm.jsonl
      echo "cleaning done after $(($(date +%s) - START))s"
    else
      echo "cleaning failed: using the original questions"
    fi
  fi
  if [ "$TU_BAT" = "1" ] && [ -z "${DATAMATE_NORM_URL:-}" ] && [ -d "$NORM_MODEL" ]; then
    stop_pids "${NPIDS[@]}"
    start_9b
    # 9B did not come back on any GPU (memory not yet released...): wait a bit and start again, otherwise the whole set is left blank
    if [ -z "$LIVE" ]; then echo "9B not back yet, trying again"; stop_all; sleep 30; start_9b; fi
    if [ -z "$LIVE" ]; then echo "ERROR: 9B could not come back after the cleaning step!" >&2; exit 1; fi
    export DATAMATE_LLM_URL="$LIVE"
    echo "9B restarted after $(($(date +%s) - START))s: $LIVE"
  fi
fi

# 2. questions: the LLM composes a JSON plan / writes SQL; the code runs it, votes, formats per §9
# At most 3 rounds: a question with a network / server error and no line yet is redone in the next round (questions with a line are skipped), while time remains
NQ=$(grep -c . "$Q")
if [ -n "$DATAMATE_LLM_URL" ] && [ -n "$DB" ]; then
  for LUOT in 1 2 3; do
    LEFT=$((BUDGET - ($(date +%s) - START))); [ "$LEFT" -lt 60 ] && break
    $PY -m datamate.query.answer "$QSPLIT" "$DB" --out "$OUT" --workers "$WORKERS" --han-giay "$LEFT" \
        --examples "$EX_TRAIN" --luat-examples "$EX_LUAT" --sql-examples "$EX_SQL"
    CO=$(grep -c . "$OUT"); echo "round $LUOT: $CO / $NQ questions have a line after $(($(date +%s) - START))s"
    [ "$CO" -ge "$NQ" ] && break
  done
fi

# rewrite the submission and log files in question order: a question with no line yet (an unexpected error) is left blank, a half-written line is dropped
$PY - "$Q" "$OUT" "$LOG_OUT" <<'EOF'
import json, os, sys
ids = [json.loads(l)["id"] for l in open(sys.argv[1], encoding="utf-8-sig") if l.strip()]
recs = {}
for l in open(sys.argv[2], encoding="utf-8"):
    try:
        r = json.loads(l)
        recs.setdefault(r["id"], r)
    except (ValueError, KeyError, TypeError):
        pass
with open(sys.argv[2] + ".tmp", "w", encoding="utf-8") as fh:
    for i in ids:
        fh.write(json.dumps(recs.get(i, {"id": i, "type": "abstain"}), ensure_ascii=False) + "\n")
os.replace(sys.argv[2] + ".tmp", sys.argv[2])

logs = {}
if os.path.exists(sys.argv[3]):
    for l in open(sys.argv[3], encoding="utf-8"):
        try:
            r = json.loads(l)
            logs.setdefault(r["id"], r)
        except (ValueError, KeyError, TypeError):
            pass
with open(sys.argv[3] + ".tmp", "w", encoding="utf-8") as lfh:
    for i in ids:
        rec = recs.get(i, {"id": i, "type": "abstain"})
        ans_obj = {k: v for k, v in rec.items() if k != "id"}
        default_log = {"id": i, "answer": ans_obj, "elapsed_ms": 0, "thinking": None, "code": None, "output": None}
        lfh.write(json.dumps(logs.get(i, default_log), ensure_ascii=False) + "\n")
os.replace(sys.argv[3] + ".tmp", sys.argv[3])
print("all", len(ids), "questions, answered", sum(recs.get(i, {}).get("type", "abstain") != "abstain" for i in ids))
EOF
echo "xong sau $(($(date +%s) - START))s"
