# docker/

Packages and runs the whole pipeline: build each shop's database, clean the questions, answer them, write the submission file. Every step runs offline,
and the models are served by vLLM inside the container.

| File | Purpose |
|---|---|
| `run.sh` | The whole procedure (described below). Works inside and outside Docker |
| `predict.sh` | The contest's fixed entry point `/code/predict.sh`; it only calls `run.sh` |
| `Dockerfile` | Image based on `vllm/vllm-openai:v0.17.0`, installs `duckdb==1.5.6`, `pandas==2.3.3`, `python-calamine==0.8.3` |
| `Dockerfile.dockerignore` | Keeps `.git` and `__pycache__` out of the build context |

> This Dockerfile was rewritten for the public repo and has **not been rebuilt**; the image submitted to the contest was built another way.

## What `run.sh` does
1. **Find the question set**: `questions.jsonl` under `/data`; the shop folder is the `data/` next to it.
2. **Start vLLM** with Qwen3.5-9B, one server per GPU (port `8000 + GPU index`). A server that does not come up is restarted once with a lighter configuration.
3. **Build the DBs** (`datamate.run_all`): read the files, assign canonical column names with the 9B model, write one DuckDB per shop.
4. **Clean the questions** (`datamate.query.normalize_all`): stop 9B, start Qwen3.5-4B, fix abbreviations, missing accents, dialect words; with a safety check,
   a question that fails keeps its original text. Then stop 4B and restart 9B.
5. **Answer** (`datamate.query.answer`, policy `ca_hai`): the JSON tier and the SQL tier, at most 3 rounds for questions with no line yet.
6. **Write the results** in question order; a missing question is written as `abstain`.

## Prepare (not in the repo)
| Needed | Place at | Notes |
|---|---|---|
| Qwen3.5-9B (LoRA merged if used) | `docker/models/qwen/` | Has `config.json` and the safetensors weights. Base model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache-2.0 |
| Qwen3.5-4B (LoRA merged if used) | `docker/models/qwen_norm4b/` | Base model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Apache-2.0 |
| Few-shot example store | `docker/assets/` | `examples_train.jsonl`, `examples_luat.jsonl`, `examples_sql_train.jsonl`, their `*_norm.jsonl` versions, and an optional `readme_train/` folder. Formats and how to build them: [next section](#the-example-store-dockerassets) |

`docker/models/` and `docker/assets/` are ignored by `.gitignore`. If `examples_train_norm.jsonl` is missing, the question-cleaning step is skipped.

## The example store (`docker/assets/`)
The answering tiers show the model a few solved questions of other shops (few-shot). Those examples live in `docker/assets/`. They come from the contest's training set,
which is not in the repo, so you have to build the store yourself. Every file is JSON Lines, UTF-8, one example per line. The lines below are **invented** to show the shape.

| File | One line holds | Used by |
|---|---|---|
| `examples_train.jsonl` | a training question and the JSON plan that answers it | JSON tier (`ExampleBank` in `query/compose.py`) |
| `examples_sql_train.jsonl` | the same question with the SQL that answers it | SQL tier (`SqlBank` in `query/sql_tier.py`) |
| `examples_luat.jsonl` | a rule line of a shop's rulebook, a question that uses it, and the plan and SQL that read the rule correctly | both tiers, to show how an unfamiliar definition is read (`RuleBank`) |
| `examples_*_norm.jsonl` | the same lines with the question rewritten by the 4B cleaning step | used instead of the raw files when the questions were cleaned |
| `readme_train/<shop>/README.md` | the training shops' `README.md` files | `--base-readmes` of `datamate.run_all`: adds column names the question set's own READMEs never described |

`examples_train.jsonl`:
```json
{"id": "q_0001", "shop": "biz_001", "as_of_date": "2026-04-15", "question": "Doanh thu quý 1 năm 2026 là bao nhiêu?", "kind": "revenue",
 "plan": {"nguon": "dong_ban", "phep": "tong", "cot": "tien", "ky": {"loai": "khoang", "tu": "2026-01-01", "den": "2026-03-31"}}, "want": null, "type": "number"}
```
`kind` is the question form, `plan` the JSON plan (keys in the [glossary](../README.md#glossary-of-vietnamese-terms-in-the-code)), `type` the answer type of the training answer, `want` the type to force (usually `null`).

`examples_sql_train.jsonl` has the same fields plus `sql` (one DuckDB statement over the `v_*` views) and `ra` (the answer type the SQL returns):
```json
{"id": "q_0001", "shop": "biz_001", "as_of_date": "2026-04-15", "question": "Doanh thu quý 1 năm 2026 là bao nhiêu?", "kind": "revenue", "type": "number",
 "plan": {"nguon": "dong_ban", "phep": "tong", "cot": "tien", "ky": {"loai": "khoang", "tu": "2026-01-01", "den": "2026-03-31"}},
 "sql": "SELECT SUM(tien) FROM v_dong_ban WHERE hop_le AND ngay BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'", "ra": "number"}
```

`examples_luat.jsonl`: `dong` identifies the rule line, `rule` is its text, `cach_tinh` a one-sentence reading of it, `sql` and `ra_sql` the SQL tier's version (a rule line may have a plan only or SQL only):
```json
{"id": "l_0001", "shop": "biz_002", "dong": "r_12", "as_of_date": "2026-04-15", "rule": "| Doanh thu | tổng tiền các đơn hoàn thành |",
 "question": "Doanh thu quý 1 năm 2026?", "type": "number", "cach_tinh": "tổng tiền đơn hoàn thành trong quý 1",
 "plan": {"nguon": "dong_ban", "phep": "tong", "cot": "tien", "ky": {"loai": "khoang", "tu": "2026-01-01", "den": "2026-03-31"}},
 "sql": "SELECT SUM(tien) FROM v_dong_ban WHERE hop_le", "ra_sql": "number"}
```
The example banks pick the lines whose question skeleton (names and numbers masked) is closest to the question being asked, never from the same shop. The five shops in `HOLDOUT`
(`query/evaluate.py`) are skipped when a store is loaded: they are kept for measuring.

**How to build it** (each step needs the training question set, with `answers.jsonl`; not run end to end for this README):
```bash
# 1. databases of the training shops
python -m datamate.run_all <train set>/data --out <DB> --skip-tables
# 2. plans: translate each question form the SQL templates understand into a plan, keep the ones whose result equals the training answer
python -m datamate.query.regress <train set> <DB>/<train set name> --out examples_train.jsonl
# 3. SQL: translate each kept plan into one statement, rerun it, keep the ones that still match the answer
python -m datamate.query.plan_sql <train set> <DB>/<train set name> --examples examples_train.jsonl --out examples_sql_train.jsonl
# 4. cleaned versions (a 4B server must be running, see DATAMATE_NORM_URL below)
python -m datamate.query.normalize_all --input examples_train.jsonl --output examples_train_norm.jsonl --db-root <DB>/<train set name>
```
The store therefore only holds the question forms the templates cover; the rest of the questions rely on the model reading the rulebook.
`examples_luat.jsonl` (and so `examples_luat_norm.jsonl`) **cannot be rebuilt from this repo**: the team wrote the rule lines and their plans by hand, and the code that produced them is not included.
Answering still runs without it: `answer.py` skips a missing `examples_luat.jsonl` or `examples_sql_train.jsonl` and the tiers get fewer examples. `examples_train.jsonl` itself is required, and without `examples_train_norm.jsonl` `run.sh` skips the cleaning step.

## Build and run
Build from the **repo root**:
```bash
docker build -f docker/Dockerfile -t datamate .
```
Run:
```bash
docker run --rm --gpus all --network none --shm-size=16g \
  -v /path/to/question-set:/data:ro -v /path/to/results:/result \
  datamate
```
The question-set folder needs `questions.jsonl`, `rulebook.md`, `calendar.csv` and a `data/<shop id>/` folder (the ledger files and `README.md` of each shop).
Output: `/result/submission.jsonl` (one line per question, in question order) and `/result/logs/logs.jsonl` (answer, time, code that ran).

For several GPUs use `--gpus all` or `--gpus '"device=0,1"'`; each GPU runs one server. The GPU count comes from `nvidia-smi`, override it with `NGPU`.

## Environment variables
| Variable | Default | Meaning |
|---|---|---|
| `BUDGET` | `19800` | Maximum seconds from the start; past it, questions not yet done are written as `abstain` |
| `GPU_MEM` | `0.80` | Share of GPU memory for the vLLM 9B server. A 24 GB GPU needs `0.92`, otherwise the KV cache is negative and the server does not start |
| `WORKERS` | `32` | Number of questions run in parallel |
| `NORM` | `1` | `0` turns off the 4B cleaning step |
| `NORM_MODEL` | `/models/qwen_norm4b` | Path of the 4B model |
| `NORM_WORKERS` | `32` | Threads for question cleaning |
| `MODEL` | `/models/qwen` | Path of the 9B model |
| `NGPU` | visible GPU count | Number of vLLM servers |
| `BATCH_INVARIANT` | `0` | `1` turns on `VLLM_BATCH_INVARIANT` (beta, slower) |
| `PREFIX_CACHE` | `0` | `1` turns on prefix caching (Qwen3.5 has linear-attention layers, still experimental) |
| `APP`, `DATA`, `RESULT`, `WORK` | `/app`, `/data`, `/result`, `/tmp/datamate` | Change the folders when running outside Docker |
| `DATAMATE_*` | see below | Settings of the pipeline code itself; `run.sh` sets some of them |

LLM sampling is fixed (`seed=2026`, `PYTHONHASHSEED=0`, single-threaded DuckDB) so a rerun gives nearly the same results.

## `DATAMATE_*` variables
The Python code reads these from the environment. A variable shown as "default on" is on unless set to the "off" value; "default off" means it only
acts when set. `run.sh` sets the ones marked *(run.sh)*.

**Model servers**
| Variable | Default | Meaning |
|---|---|---|
| `DATAMATE_LLM_URL` | empty | OpenAI-compatible server(s) of the 9B model, comma-separated for several. Empty: the loader runs on rules only and `--llm` has no effect. When set before `run.sh`, `run.sh` does **not** start vLLM *(run.sh sets it to the servers it starts)* |
| `DATAMATE_LLM_KEY` | empty | API key; one per URL, comma-separated *(run.sh: `none`)* |
| `DATAMATE_LLM_MODEL` | empty | Model name sent in each request *(run.sh: `qwen`)* |
| `DATAMATE_LLM_EXTRA` | `{}` | JSON object merged into every request, e.g. `{"reasoning_effort": "none"}` for a server that turns Qwen3.5's thinking mode on by itself *(run.sh sets the sampling fields `top_p`, `top_k`, `min_p`, `repetition_penalty`)* |
| `DATAMATE_LLM_CACHE` | empty | A `.jsonl` file that stores answers by (model, prompt), so a rerun does not ask the model again |
| `DATAMATE_HAN_CHOT` | `0` (no deadline) | Unix time of the deadline. Near it each call shortens its wait; past it no more calls are made *(run.sh: start time + `BUDGET`)* |
| `DATAMATE_NORM_URL` | `http://127.0.0.1:8001` for `normalize_all`; empty for `run.sh` | The 4B server used to clean questions. Set before `run.sh`, it uses that server and does not start its own |
| `DATAMATE_NORM_KEY` | empty | API key of the 4B server |
| `DATAMATE_NORM_MODEL` | `qwen_norm` | Model name of the 4B server. Not the same as `NORM_MODEL` above, which is a folder |

**Answering**
| Variable | Default | Meaning |
|---|---|---|
| `DATAMATE_POLICY` | `ca_hai` | How the JSON and SQL tiers are combined (`--policy` of `datamate.query.answer`). `ca_hai`: ask both, see [How it works](../README.md#how-it-works). `json` or `sql`: one tier only. `dpc`: five SQL candidates are clustered and cross-checked with the JSON plan (`dpc_select`). The other values (`dinh_tuyen`, `bi_thi_sql`, `bi_thi_json`, `doi_chieu`) are variants used for comparison; see `solve_both` in `datamate/query/hybrid.py` |
| `DATAMATE_TELEX` | on | `0` turns off Telex typo repair (`thangs` → `tháng`) on questions. `run.sh` sets `1`, then `0` once the 4B cleaning step has produced cleaned questions |
| `DATAMATE_LUAT_GOC` | on | `0` turns off the check that blanks an answer which relies on a built-in rule the contest set has redefined |
| `DATAMATE_LUAT_DONG` | off | `1` also looks for redefined rules in the `§6` section and in the shop's `README.md` bullets (many false alarms in training, hence off) |

**Data layer: switches on by default in `run.sh`**
| Variable | Meaning |
|---|---|
| `DATAMATE_NV2` | Debt can be computed when the order table has no order total *(run.sh: `1`)* |
| `DATAMATE_TON_DONG` | Stock = purchases − shipments + returns *(run.sh: `1`)* |
| `DATAMATE_TU_SUA` | The self-repair loop (`loader/tu_sua.py`): after the build, the LLM proposes how to re-read a ledger that looks wrong; a proposal is kept only if the checks improve. Needs `--llm`. Off when the variable is unset *(run.sh: `1`)* |

**Data layer: switches on by default** (set the variable to `0` to turn off)
| Variable | Meaning |
|---|---|
| `DATAMATE_POS_PHANG` | SQLite files from point-of-sale software: resolve numeric ids to codes and names, Unix time to Vietnam time |
| `DATAMATE_SO_NGAY` | A sheet that groups rows under day headings (a row holding only a date) gets a date on every row; an auxiliary ledger that numbers orders per day gets order ids of the form `dd/mm/yyyy-n` |
| `DATAMATE_DAY_FILL` | A row with an empty date takes the date of the nearest row above it, when the ledger writes the date once per day |
| `DATAMATE_HDR` | Order-header ledgers: detect the order id from the other table's order codes |
| `DATAMATE_TT_SO_PHU` | Order status from an auxiliary ledger (cancelled, returned, not yet delivered); an order not in it is completed |
| `DATAMATE_GOP_NGUON` | Merge the real sales ledgers of several sources (counter, marketplace, online) into one sales table, and build order ids from serial numbers when a ledger has none |
| `DATAMATE_TEN_THO` | Replace a worker's short name by the full name when it matches exactly one person |
| `DATAMATE_KHOP_TEN` | Match product names in sales and purchase rows that are in no catalogue to catalogue names (adds aliases) |

**Data layer: switches off by default** (set the variable to `1` to turn on; the reason each is off is in the code comment next to it)
| Variable | Meaning |
|---|---|
| `DATAMATE_NV3` | A table named like a receipt or payment ledger (order id, money, date, no product columns): its date and amount become `payment_date` and `payment_amount` |
| `DATAMATE_NV4` | Add more built-in column names (base unit, order total, member code, payment method); changes the LLM's column-matching prompt |
| `DATAMATE_INLINE` | Read column descriptions written inline in a shop's `README.md` |
| `DATAMATE_KHACH_KHONG_LAP` | A customer list whose codes repeat a lot is treated as an event ledger |
| `DATAMATE_QUY_DOI_GIA` | Guess unit conversions from price ratios (it can mistake different pack sizes for a conversion) |

**Training scripts**
| Variable | Default | Meaning |
|---|---|---|
| `DATAMATE_ROOT` | `/app` | Folder that holds the `datamate` package, for `finetune/qwen4b_norm/build_norm_v2_dataset_ptv2.py` and `finetune/qwen9b/cham_kiem_ft.py`. Outside Docker, set it to the repo root |
