#!/usr/bin/env bash
# Stand-in for CryoZeta's inference_demo.sh.
#
# Accepts the same flags, emits log lines that match the real script's stage
# markers, and writes an output tree with the same shape -- so the web/job
# workflow can be exercised end to end without a GPU.
#
# Test hooks (environment variables):
#   FAKE_EXIT_CODE   exit with this code instead of 0
#   FAKE_SLEEP       seconds to sleep mid-run (for cancellation tests)
#   FAKE_STDERR      write this message to stderr
set -e

input_json=""
output_dir=""
gpu=""
mode="combined"
env_name=""
overwrite="false"
example=""
registration=""

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--input-json)  input_json="$2"; shift 2 ;;
        -o|--output-dir)  output_dir="$2"; shift 2 ;;
        -g|--gpu)         gpu="$2"; shift 2 ;;
        -m|--mode)        mode="$2"; shift 2 ;;
        -e|--env)         env_name="$2"; shift 2 ;;
        -x|--example)     example="$2"; shift 2 ;;
        -r|--registration) registration="$2"; shift 2 ;;
        --overwrite)      overwrite="true"; shift ;;
        *) echo "fake: unknown option '$1'" >&2; exit 64 ;;
    esac
done

[ -n "$input_json" ] || { echo "fake: missing --input-json" >&2; exit 64; }
[ -n "$output_dir" ] || { echo "fake: missing --output-dir" >&2; exit 64; }
[ -f "$input_json" ] || { echo "fake: input json not found: $input_json" >&2; exit 65; }

echo "==> fake CryoZeta"
echo "==> env=${env_name} gpu=${gpu} mode=${mode} overwrite=${overwrite}"
echo "==> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Pull the entry name out of the generated JSON, and assert its shape while
# we are at it: this is what makes the integration test a real contract check.
name=$(python3 - "$input_json" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
assert isinstance(data, list) and len(data) == 1, "input JSON must be a 1-entry list"
entry = data[0]
for key in ("name", "map_path", "resolution", "contour_level", "sequences"):
    assert key in entry, f"missing key: {key}"
assert entry["contour_level"] != 0, "contour_level must be non-zero"
assert entry["sequences"], "no sequences"
for block in entry["sequences"]:
    assert len(block) == 1, "each sequence entry has exactly one chain-type key"
    (chain_type, body), = block.items()
    assert chain_type in ("proteinChain", "dnaSequence", "rnaSequence"), chain_type
    assert "sequence" in body and "count" in body
    if chain_type == "dnaSequence":
        assert "msa" not in body, "DNA must not carry an msa block"
    else:
        assert "precomputed_msa_dir" in body["msa"], "protein/RNA need an MSA dir"
print(entry["name"])
PY
) || { echo "fake: input JSON failed validation" >&2; exit 66; }

echo "==> entry: ${name}"

if [ -n "${FAKE_STDERR:-}" ]; then echo "$FAKE_STDERR" >&2; fi

# --- stage 1: detection -------------------------------------------------
echo "==> Running detection to generate EM .pt for selected sample..."
mkdir -p "${output_dir}/${name}/CryoZeta-Detection"
echo "detection 12.5s" > "${output_dir}/${name}/CryoZeta-Detection/${name}_timing.txt"
touch "${output_dir}/${name}/CryoZeta-Detection/${name}.pt"

if [ -n "${FAKE_SLEEP:-}" ]; then sleep "$FAKE_SLEEP"; fi

if [ -n "$example" ]; then
    # --- large / cycle path --------------------------------------------
    echo "==> Starting large complex cycle prediction..."
    echo "==> cryozeta-cycle-predict"
    echo "==> Combining stages into final structure..."
    printf 'data_%s\n#\n' "$name" > "${output_dir}/combined.cif"
else
    # --- standard path --------------------------------------------------
    seed_dir="${output_dir}/${name}/CryoZeta/seed_101/predictions"
    mkdir -p "$seed_dir" "${output_dir}/${name}/CryoZeta/saved_data"
    if [ "$mode" = "combined" ] || [ "$mode" = "cryozeta" ]; then
        echo "==> cryozeta-inference --use_interpolation false"
    fi
    if [ "$mode" = "combined" ] || [ "$mode" = "cryozeta-interpolate" ]; then
        echo "==> cryozeta-inference --use_interpolation true"
        mkdir -p "${output_dir}/${name}/CryoZeta-Interpolate/seed_101/predictions"
    fi

    for i in 0 1 2; do
        printf 'data_%s_sample_%s\n#\n' "$name" "$i" > "${seed_dir}/${name}_sample_${i}.cif"
        printf '{"ptm": 0.8%s, "iptm": 0.7%s}\n' "$i" "$i" \
            > "${seed_dir}/${name}_summary_confidence_sample_${i}.json"
    done
    printf 'sample,method,recall_ccmask_ca\n0,teaser,0.91\n1,svd,0.88\n' \
        > "${output_dir}/${name}/CryoZeta/saved_data/scores.csv"

    if [ "$mode" = "combined" ]; then
        echo "==> cryozeta-combine"
        final="${output_dir}/${name}/CryoZeta-Final"
        mkdir -p "$final"
        for i in 0 1 2; do
            printf 'data_%s_final_%s\n#\n' "$name" "$i" > "${final}/${name}_sample_${i}.cif"
        done
    fi
fi

echo "==> Done!"
exit "${FAKE_EXIT_CODE:-0}"
