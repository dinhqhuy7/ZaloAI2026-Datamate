#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00
#SBATCH -J dm_gop_mix
# Merge the round-2 adapter into the base model -> a folder to place at docker/models/qwen. The adapter that was used is kept in adapter_mix/.
W=$(cd "$(dirname "$0")" && pwd); BASE=${BASE:-models/Qwen3.5-9B}; PY=${PY:-/opt/train_venv/bin/python3}; [ -x "$PY" ] || PY=python3
LORA=${LORA:-$W/runmix/lora}
if [ ! -d "$LORA" ] && [ "${USE_PACKAGED_ADAPTER:-0}" = "1" ]; then LORA=$W/adapter_mix; fi
[ -d "$LORA" ] || { echo "ERROR: LoRA not found at $LORA" >&2; exit 1; }
$PY "$W/gop_lora.py" "$BASE" "$LORA" "${OUT:-qwen9b_mix_merged}" && echo GOP_XONG
