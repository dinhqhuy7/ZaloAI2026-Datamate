# Training the two models used at run time

The two LoRA adapters below were merged into the `Qwen3.5-9B` and `Qwen3.5-4B` weights that the pipeline uses at run time. This folder holds the
code to check or redo them; the `.jsonl` data and adapters named below are **not in the repo** (see `README.md`).
This folder is not used when the pipeline runs.

## qwen9b/: Qwen3.5-9B, LoRA for the plan tier (JSON + SQL)
- Training: QLoRA 4-bit (nf4, compute bf16), r = 16, alpha = 32, dropout 0.05, every linear layer (except the image / video part,
  mtp, lm_head); the loss is computed only on the answer part. One 24 GB GPU (RTX 3090 Ti), Python 3.12, torch 2.11 (cu128),
  transformers 5.2x, peft 0.21.
- Round 1 (`train_800.sh`): `thu_train.jsonl` (800 JSON plan samples), evaluated on `thu_val.jsonl`.
- Round 2 (`train_mix.sh`): continue from the round-1 adapter on `mix_train.jsonl` (2,200 samples: 700 JSON plans, 1,100 SQL,
  400 rulebook rule-change samples), 1 epoch. `sql_gen_train.jsonl`, `sql_pv2.jsonl`, `rbd_sql_A.jsonl`, `rbd_sql_C.jsonl` are the sources
  that were combined into the SQL part of the mix.
- Merge (`gop_mix.sh` -> `gop_lora.py`): W += (alpha / r) * B @ A for each matrix that has LoRA. The adapter that was used: `adapter_mix/`.
- Data creation: `make_plan_data.py`, `make_sql_data.py` (they need the `datamate/` code and the generated shop sets that have solutions).
  The prompt in the data is exactly the prompt the pipeline sends at run time. `test_shops.txt`: shops held out for evaluation, not trained on.
- Evaluation: `eval_kehoach.py`, `cham_kiem_ft.py`.
- Data sources: the contest organiser's (BTC) train set and the team's generated shop sets (shops, questions, answers produced by the generator).

## qwen4b_norm/: Qwen3.5-4B, LoRA for question cleaning
- Data: `build_norm_v2_dataset_ptv2.py --cleaned` -> `full_gold_norm_v2_chatml_cleaned.jsonl` (4,669 rows).
  Sources: BTC train questions (`biz_101`..`biz_130`), the generated set `bo_tu_sinh_v2` (`biz_501`..`biz_536`: noisy question -> clean question produced by the generator), rule samples from the train rulebook, synthetic samples (rule changes, new industries, keeping an already clean question).
- `A_train_lora_unsloth.ipynb` (Unsloth, bf16, max_seq_length 768) on `full_gold_norm_v2_chatml_cleaned.jsonl`.
- The adapter that was used: `adapter_v2/`; merged with `gop_lora.py` as above.

Environment variables of the scripts: `BASE` (base model folder), `PY` (a Python with all libraries), `LORA`, `OUT`.
