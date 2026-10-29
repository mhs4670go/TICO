#!/usr/bin/env bash
# Reproduce exports in separate processes; no benchmark runs in this script.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "Usage: PYTHON_BIN=/path/to/venv/python $0 BASE_TREE A_TREE B_TREE OUTPUT_DIR" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON_BIN:-python}
base_tree=$(realpath "$1")
a_tree=$(realpath "$2")
b_tree=$(realpath "$3")
output_dir=$(realpath -m "$4")
mkdir -p "$output_dir"
for variant in base a b; do
  case "$variant" in
    base) tree=$base_tree ;;
    a) tree=$a_tree ;;
    b) tree=$b_tree ;;
  esac
  (
    cd "$tree"
    export PYTHONPATH="$tree"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    mkdir -p "$output_dir/$variant"
    "$python_bin" -c 'import pathlib,tico; print(tico.__file__); assert pathlib.Path(tico.__file__).resolve().parent.parent==pathlib.Path.cwd().resolve()' > "$output_dir/$variant/import.log"
    "$python_bin" -m tico.quantization.examples.inspector --mode wrapper-smoke --list-cases > "$output_dir/$variant/cases.log"
    for mode in prefill decode; do
      "$python_bin" -m tico.quantization.examples.inspector \
        --config tico/quantization/examples/configs/wrapper_smoke.yaml \
        --mode wrapper-smoke --case "gemma4_text_attention_$mode" \
        --export circle --output-dir "$output_dir/$variant/inspector-$mode" \
        --no-plot --strict --report-json "$output_dir/$variant/inspector-$mode.json" \
        > "$output_dir/$variant/inspector-$mode.log" 2>&1
    done
    "$python_bin" "$script_dir/export_check.py" --output "$output_dir/$variant/custom" > "$output_dir/$variant/custom.log" 2>&1
    "$python_bin" "$script_dir/export_check.py" --hq 4 --hkv 2 --output "$output_dir/$variant/tiny" > "$output_dir/$variant/tiny.log" 2>&1
    "$python_bin" "$script_dir/inspector_runtime.py" --artifact-dir "$output_dir/$variant" > "$output_dir/$variant/inspector-runtime.log" 2>&1
    "$python_bin" "$script_dir/export_identity.py" --output "$output_dir/$variant/identity.json"
  )
done
"$python_bin" "$script_dir/summarize_export.py" --artifact-root "$output_dir" --output "$output_dir/summary"
# Baseline FP runtime shape failures are recorded in JSON; review per-stage status.
