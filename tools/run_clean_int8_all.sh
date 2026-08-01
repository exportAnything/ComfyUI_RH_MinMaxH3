#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/MiniMax-H3" >&2
    exit 2
fi

BASE="$(readlink -f "$1")"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE/.cache/h3_quantization"
LOG_FILE="$LOG_DIR/clean_int8_${RUN_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_exit() {
    code=$?
    echo "[$(date -Is)] quantization exit=$code"
    exit "$code"
}
trap on_exit EXIT

check_source() {
    local component="$1"
    [[ -f "$component/config.json" ]] || {
        echo "missing config: $component/config.json" >&2
        exit 3
    }
    compgen -G "$component/*.safetensors" >/dev/null || {
        echo "missing safetensors: $component" >&2
        exit 3
    }
}

check_absent() {
    local path="$1"
    [[ ! -e "$path" ]] || {
        echo "refusing to overwrite existing output: $path" >&2
        exit 4
    }
}

validate_meta() {
    local output_dir="$1"
    local expected="$2"
    python3 -c '
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
meta = json.loads((p / "quant_meta.json").read_text())
actual = int(meta["quantized_linears"])
if actual != expected:
    raise SystemExit(f"quantized_linears mismatch: expected={expected}, actual={actual}, path={p}")
weights = list(p.glob("*.safetensors"))
if len(weights) != 1 or weights[0].stat().st_size <= 0:
    raise SystemExit(f"invalid output safetensors: {weights}")
print(f"validated {p}: linears={actual}, file={weights[0].name}, bytes={weights[0].stat().st_size}")
' "$output_dir" "$expected"
}

FL="$BASE/FL2VA"
REF="$BASE/Ref2VA"
FL_TE_BUILD="$FL/.text_encoder_int8_convrot.building-$RUN_ID"
REF_TE_BUILD="$REF/.text_encoder_int8_convrot.building-$RUN_ID"
FL_DIT_BUILD="$FL/.transformer_int8_convrot.building-$RUN_ID"
REF_DIT_BUILD="$REF/.transformer_int8_convrot.building-$RUN_ID"

check_source "$FL/text_encoder"
check_source "$FL/transformer"
check_source "$REF/transformer"
for path in \
    "$FL/text_encoder_int8_convrot" "$REF/text_encoder_int8_convrot" \
    "$FL/transformer_int8_convrot" "$REF/transformer_int8_convrot" \
    "$FL_TE_BUILD" "$REF_TE_BUILD" "$FL_DIT_BUILD" "$REF_DIT_BUILD"; do
    check_absent "$path"
done

echo "[$(date -Is)] clean official-source INT8 quantization started"
echo "base=$BASE"
echo "log=$LOG_FILE"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python3 tools/quantize_text_encoder_int8_convrot.py \
    --src "$FL/text_encoder" \
    --dst "$FL_TE_BUILD" \
    --device cuda \
    --verify
validate_meta "$FL_TE_BUILD" 350
mv "$FL_TE_BUILD" "$FL/text_encoder_int8_convrot"

cp -a "$FL/text_encoder_int8_convrot" "$REF_TE_BUILD"
validate_meta "$REF_TE_BUILD" 350
mv "$REF_TE_BUILD" "$REF/text_encoder_int8_convrot"

python3 tools/quantize_int8_convrot.py \
    --src "$FL/transformer" \
    --dst "$FL_DIT_BUILD" \
    --partition FL2VA \
    --device cuda \
    --verify \
    --verify-report "$FL_DIT_BUILD/verify_report.tsv"
validate_meta "$FL_DIT_BUILD" 201
mv "$FL_DIT_BUILD" "$FL/transformer_int8_convrot"

python3 tools/quantize_int8_convrot.py \
    --src "$REF/transformer" \
    --dst "$REF_DIT_BUILD" \
    --partition Ref2VA \
    --device cuda \
    --verify \
    --verify-report "$REF_DIT_BUILD/verify_report.tsv"
validate_meta "$REF_DIT_BUILD" 201
mv "$REF_DIT_BUILD" "$REF/transformer_int8_convrot"

sync
echo "[$(date -Is)] CLEAN_INT8_ALL_COMPLETE"
