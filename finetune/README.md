# finetune/ — data generation and fine-tuning

Code that generates synthetic data and trains the two LoRA adapters used by the pipeline (`datamate/`, `docker/`) at run time. This folder is **not** needed when the pipeline runs:
the two adapters are already merged into the `Qwen3.5-9B` and `Qwen3.5-4B` weights.

## Structure
| Folder / file | Purpose |
|---|---|
| `bo_sinh/` | Synthetic shop generator: ledgers, READMEs, questions, correct answers, JSON-plan solutions. Run with `sinh.py`; `thu_vien_goc/` is the simulation library (core, sim, engine, qgen, writers); `nganh*.py` are industry configurations; `luat_doi.py` generates rulebooks with changed rules |
| `qwen9b/` | Qwen3.5-9B: data creation (`make_plan_data.py`, `make_sql_data.py`), 2-round QLoRA training (`train_kehoach.py`, `train_800.sh`, `train_mix.sh`), LoRA merging (`gop_lora.py`, `gop_mix.sh`), evaluation (`eval_kehoach.py`, `cham_kiem_ft.py`), `test_shops.txt` (held-out shops, not trained on) |
| `qwen4b_norm/` | Qwen3.5-4B question cleaning: data building (`build_norm_v2_dataset_ptv2.py`), training (`train_4b.py` or the Unsloth notebook `A_train_lora_unsloth.ipynb`), LoRA merging (`gop_lora.py`) |
| `train.sh` | Run 9B (two rounds + merge) then 4B (training + merge) |
| `configs/` | Hyperparameters of each training run and the generator configuration (a record of the real runs; paths in them are the machine's at run time) |
| `logs/` | The real logs of generation, training, merging (with `trainer_state.json`) |
| `requirements.txt` | Training libraries |
| `GUIDE.md` | A short description of each model: how it was trained, the data, the evaluation |

## The two models
| Base model | Revision | Role in the pipeline | Training |
|---|---|---|---|
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Build the DB, JSON plans, write SQL | QLoRA 4-bit (nf4, bf16), r=16, alpha=32, dropout 0.05; 2 rounds; merge `W += (alpha/r)·B·A` |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Question cleaning | LoRA bf16, r=16, alpha=32, 3 epochs; merged the same way |

## Data (information only, no files)
| Set | Source | Samples |
|---|---|---|
| 9B round 1: `thu_train.jsonl` / `thu_val.jsonl` | generated shops (`bo_sinh/`), JSON plans | 800 / 40 |
| 9B round 2: `mix_train.jsonl` | 700 JSON plans, 1,100 SQL, 400 rulebook rule-change samples | 2,200 |
| Sources of the SQL part: `sql_gen_train.jsonl`, `sql_pv2.jsonl`, `rbd_sql_A.jsonl`, `rbd_sql_C.jsonl` | generated | 3,040 / 1,303 / 231 / 159 |
| 4B: `full_gold_norm_v2_chatml_cleaned.jsonl` / `val_norm_v2.jsonl` | BTC train questions (`biz_101` to `biz_130`), the generated set `bo_tu_sinh_v2` (noisy question → clean question), rule samples, synthetic samples | 4,669 / 100 |

Two generated sets: `gen_train` (76 shops, 3,561 questions, `--seed 7 --start 301`) and `bo_tu_sinh_v2` (36 shops, 2,473 questions, `--seed 19 --start 501`).
The training sources are the BTC `train` set and generated data. The repo contains no question or answer of public_test or private_test.

**Two generator options imitate the form of the public question set.** `sinh.py --extra-layouts` adds three extended ledger styles (one file per month, several xlsx files of the same table, invoices only),
and `--spoken-style` dirties questions by a spoken-style distribution (for example 92% end with `?`, 6% unaccented, 38% abbreviated). According to the comments in the code (`sinh.py`, `van_public()`,
`mo_rong_1311.py`), these parameters come from form statistics observed on the public set. The `bo_tu_sinh_v2` set (36 shops, 2,473 questions) was generated with these two options, and
the log `logs/bo_sinh_generation.log` records the original flag names `--layout-public` and `--van-public`, which were renamed `--extra-layouts` and `--spoken-style` (the generated output is unchanged).

## Hardware and measured time
| Stage | Hardware | Time |
|---|---|---|
| Data generation (`bo_sinh/`) | CPU, 16 threads | about 18 minutes |
| 9B round 1 (100 steps) | 1 × RTX 3090 Ti 24 GB | 2 h 15 min (8,120 s) |
| 9B round 2 (275 steps, loss 0.064) | 1 × RTX 3090 Ti 24 GB | 7 h 08 min (25,670 s) |
| Merge LoRA into 9B (248 matrices) | CPU, 64 GB RAM | about 4 minutes |
| 4B (4,669 samples, 3 epochs) | 1 × A100 40 GB (`configs/qwen4b_norm_v2_config.json`) | about 18 minutes |

## Redoing it (steps)
```bash
# 1. Generate the synthetic data (see "Not in the repo" for the two input files)
cd finetune/bo_sinh
python3 -I sinh.py thu_vien_goc /tmp/gen_train --n 76 --seed 7 --start 301 --loi-giai
python3 -I sinh.py thu_vien_goc /tmp/bo_tu_sinh_v2 --n 36 --seed 19 --start 501 --extra-layouts --spoken-style --cau-lech --loi-giai

# 2. Training data for 9B (needs the code in datamate/ and the example store docker/assets/; set PYTHONPATH to the repo root)
cd ../qwen9b
export PYTHONPATH=$(git rev-parse --show-toplevel)
python3 make_plan_data.py --gen /tmp/gen_train --gen-db /tmp/gen_db --gen-cham /tmp/gen_cham.jsonl --out /tmp/plan_out --test-shops biz_801
python3 make_sql_data.py  --gen /tmp/gen_train --gen-db /tmp/gen_db --gen-cham /tmp/gen_cham.jsonl --out /tmp/sql_out.jsonl --test-shops biz_801

# 3. Train and merge 9B
BASE=/path/to/Qwen3.5-9B bash train_800.sh
BASE=/path/to/Qwen3.5-9B bash train_mix.sh
BASE=/path/to/Qwen3.5-9B OUT=/path/to/qwen bash gop_mix.sh

# 4. 4B: build the data, train, merge
cd ../qwen4b_norm
python3 build_norm_v2_dataset_ptv2.py --cleaned
python3 train_4b.py --model /path/to/Qwen3.5-4B --train full_gold_norm_v2_chatml_cleaned.jsonl --val val_norm_v2.jsonl --out adapter_v2
python3 gop_lora.py /path/to/Qwen3.5-4B adapter_v2 /path/to/qwen_norm4b
```
`train.sh` combines steps 3 and 4 (`BASE`, `BASE_4B`, `PY` set through environment variables). Variables shared by the scripts: `BASE`, `PY`, `LORA`, `OUT`.
The commands in steps 1 and 2 are written exactly as in the real run; `--test-shops biz_801` comes from the real run, change it to match your generator (the held-out list is in `qwen9b/test_shops.txt`).

## Seeds
`bo_sinh/sinh.py` uses `random.Random(f"{seed}-{shop id}")` (seeds 7 and 19); `build_norm_v2_dataset_ptv2.py` uses `random.Random(2026)`;
9B training uses `seed=42`, 4B uses `seed=2026`. Retraining on another GPU may give slightly different weights because of floating-point error.

## Not in the repo
- **Training data** (`*.jsonl`) and **LoRA adapters** (`adapter_mix/`, `adapter_v2/`): ignored by `.gitignore`. Redo them with the steps above.
- **The BTC `calendar.csv`**: the generator (`thu_vien_goc/core.py`) reads `bo_sinh/train/train/calendar.csv`, a solar–lunar calendar for 2024–2027 with the columns
  `date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday`. Copy this file from the BTC question set to exactly that path, otherwise `sinh.py` does not run.
  (The rulebook the generator uses to compute answers, `bo_sinh/thu_vien_goc/rulebook.md`, was written by the team and is in the repo; it follows the same section order as the BTC rulebook.)
- **The BTC train set** and the example store `docker/assets/`.
- The contest manifest (`data_manifest.json`, `make_manifest.py`): meaningful only for the submitted image.
