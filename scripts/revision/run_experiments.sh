#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

original_args=("$@")
julia_bin="${JULIA:-julia}"
mode="full"
stages="mnist,fashion-mnist,pendulum,retraction"
seeds="1234,1235,1236,1237,1238,1239,1240,1241,1242,1243"
configurations="${MNIST_CONFIGURATIONS:-all}"
output_root="${GML_RESULTS_ROOT:-$repo_root/results/revision}"
resume_dir=""
allow_dirty=0
allow_any_gpu=0
allow_no_cuda=0
retraction_repo="${GEOMETRIC_OPTIMIZERS_REPO:-$repo_root/../GeometricOptimizers}"

usage() {
    cat <<USAGE
usage: $0 [--smoke|--full] [--stages LIST] [--seeds LIST]
          [--configurations LIST] [--output-dir DIR] [--resume-dir DIR] [--allow-dirty]
          [--allow-any-gpu] [--allow-no-cuda] [--retraction-repo DIR]
USAGE
}

while (( $# )); do
    case "$1" in
        --smoke) mode="smoke"; shift ;;
        --full) mode="full"; shift ;;
        --stages) stages="${2:?missing stage list}"; shift 2 ;;
        --seeds) seeds="${2:?missing seed list}"; shift 2 ;;
        --configurations) configurations="${2:?missing configuration list}"; shift 2 ;;
        --output-dir) output_root="${2:?missing output directory}"; shift 2 ;;
        --resume-dir) resume_dir="${2:?missing run directory}"; shift 2 ;;
        --allow-dirty) allow_dirty=1; shift ;;
        --allow-any-gpu) allow_any_gpu=1; shift ;;
        --allow-no-cuda) allow_no_cuda=1; shift ;;
        --retraction-repo) retraction_repo="${2:?missing repository path}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a seed_array <<< "$seeds"
(( ${#seed_array[@]} > 0 )) || { echo "no seeds supplied" >&2; exit 2; }
if [[ "$mode" == full && ${#seed_array[@]} -ne 10 ]]; then
    echo "full mode requires exactly 10 seeds, got ${#seed_array[@]}" >&2
    exit 2
fi

if [[ -n "$resume_dir" ]]; then
    run_dir="$(cd "$(dirname "$resume_dir")" && pwd)/$(basename "$resume_dir")"
    output_root="$(dirname "$run_dir")"
else
    stamp="$(date -u +%Y%m%dT%H%M%SZ)_${mode}"
    run_dir="$output_root/$stamp"
fi
mkdir -p "$run_dir" || exit 1
[[ -w "$run_dir" ]] || { echo "output directory is not writable: $run_dir" >&2; exit 1; }
log="$run_dir/run.log"
status_file="$run_dir/stages.csv"
[[ -s "$status_file" ]] || printf 'stage,status,started_utc,finished_utc,command\n' > "$status_file"

archive_results() {
    local exit_code=$?
    set +e
    mkdir -p "$run_dir/environments/root" "$run_dir/environments/scripts"
    cp Project.toml Manifest.toml "$run_dir/environments/root/" 2>/dev/null
    cp scripts/Project.toml scripts/Manifest.toml "$run_dir/environments/scripts/" 2>/dev/null
    git rev-parse HEAD > "$run_dir/gmldatasets.sha"
    git status --porcelain=v1 > "$run_dir/gmldatasets.status"
    git diff --binary > "$run_dir/gmldatasets.patch"
    if [[ -d "$retraction_repo/.git" ]]; then
        git -C "$retraction_repo" rev-parse HEAD > "$run_dir/geometricoptimizers.sha"
        git -C "$retraction_repo" status --porcelain=v1 > "$run_dir/geometricoptimizers.status"
        git -C "$retraction_repo" diff --binary > "$run_dir/geometricoptimizers.patch"
    fi
    printf '%q ' "$0" --resume-dir "$run_dir" "${original_args[@]}" > "$run_dir/restart-command.txt"
    printf '\n' >> "$run_dir/restart-command.txt"
    tar -C "$output_root" -czf "$run_dir.tar.gz" "$(basename "$run_dir")"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$run_dir.tar.gz" > "$run_dir.tar.gz.sha256"
    else
        shasum -a 256 "$run_dir.tar.gz" > "$run_dir.tar.gz.sha256"
    fi
    echo "artifact: $run_dir.tar.gz"
    echo "checksum: $run_dir.tar.gz.sha256"
    exit "$exit_code"
}
trap archive_results EXIT

exec > >(tee -a "$log") 2>&1

if [[ "$allow_dirty" -ne 1 && -n "$(git status --porcelain=v1)" ]]; then
    echo "working tree is dirty; commit/stash changes or pass --allow-dirty" >&2
    exit 1
fi

export GML_ALLOW_ANY_GPU="$allow_any_gpu"
export GML_ALLOW_NO_CUDA="$allow_no_cuda"
export GML_REQUIRED_GPU="${GML_REQUIRED_GPU:-RTX 4090}"
"$julia_bin" --project=scripts scripts/revision/check_environment.jl > "$run_dir/environment.txt" 2>&1 || exit 1
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "$run_dir/nvidia-smi.txt" 2>&1 || exit 1
elif [[ "$allow_no_cuda" -eq 1 ]]; then
    printf '%s\n' 'nvidia-smi unavailable; CPU smoke explicitly allowed' > "$run_dir/nvidia-smi.txt"
else
    echo "nvidia-smi is unavailable" >&2
    exit 1
fi

run_stage() {
    local stage="$1"; shift
    local started finished command_text
    started="$(date -u +%FT%TZ)"
    printf -v command_text '%q ' "$@"
    echo "[$started] starting $stage"
    if "$@" > >(tee -a "$run_dir/${stage}.stdout.txt") 2> >(tee -a "$run_dir/${stage}.stderr.txt" >&2); then
        finished="$(date -u +%FT%TZ)"
        printf '%s,ok,%s,%s,"%s"\n' "$stage" "$started" "$finished" "$command_text" >> "$status_file"
    else
        local code=$?
        finished="$(date -u +%FT%TZ)"
        printf '%s,failed:%d,%s,%s,"%s"\n' "$stage" "$code" "$started" "$finished" "$command_text" >> "$status_file"
        return "$code"
    fi
}

contains_stage() { [[ ",$stages," == *",$1,"* ]]; }
repetitions="${#seed_array[@]}"
epochs=500
sae_epochs=12000
if [[ "$mode" == smoke ]]; then
    repetitions=1
    seeds="${seed_array[0]}"
    epochs=2
    sae_epochs=2
fi

run_image_dataset() {
    local dataset="$1"
    local prefix="$run_dir/$dataset"
    if [[ "$mode" == full ]]; then
        run_stage "${dataset}-warmup" env MNIST_DATASET="$dataset" MNIST_REPETITIONS=1 MNIST_SEEDS="${seed_array[0]}" MNIST_N_EPOCHS=1 MNIST_ACCURACY_EVERY=1 MNIST_CONFIGURATIONS="$configurations" MNIST_REPORT="$prefix-warmup-report.txt" MNIST_LOSSES="$prefix-warmup-losses.csv" MNIST_RECORDS="$prefix-warmup-runs.csv" MNIST_OUTPUT="$prefix-warmup.jld2" "$julia_bin" --project=scripts scripts/geometric_optimizers/mnist_cuda_repetitions.jl || return
    fi
    run_stage "$dataset" env MNIST_DATASET="$dataset" MNIST_REPETITIONS="$repetitions" MNIST_SEEDS="$seeds" MNIST_N_EPOCHS="$epochs" MNIST_CONFIGURATIONS="$configurations" MNIST_REPORT="$prefix-report.txt" MNIST_LOSSES="$prefix-losses.csv" MNIST_RECORDS="$prefix-runs.csv" MNIST_OUTPUT="$prefix.jld2" "$julia_bin" --project=scripts scripts/geometric_optimizers/mnist_cuda_repetitions.jl
}

if contains_stage mnist; then
    run_image_dataset mnist || exit $?
fi
if contains_stage fashion-mnist; then
    run_image_dataset fashion-mnist || exit $?
fi

if contains_stage pendulum; then
    records="$run_dir/pendulum-runs.csv"
    if [[ "$mode" == full ]]; then
        run_stage pendulum-warmup env SAE_REQUIRE_CUDA=1 SAE_SEED="${seed_array[0]}" SAE_N_EPOCHS=1 SAE_OUTPUT="$run_dir/pendulum-warmup.h5" "$julia_bin" --project=scripts scripts/pendulum/train_sae.jl || exit $?
    fi
    repetition=0
    for seed_value in ${seeds//,/ }; do
        repetition=$((repetition + 1))
        checkpoint="$run_dir/pendulum-seed-${seed_value}.h5"
        [[ -s "$checkpoint" ]] && { echo "skipping completed checkpoint $checkpoint"; continue; }
        require_cuda=1
        [[ "$allow_no_cuda" -eq 1 ]] && require_cuda=0
        run_stage "pendulum-seed-${seed_value}" env SAE_REQUIRE_CUDA="$require_cuda" SAE_SEED="$seed_value" SAE_REPETITION="$repetition" SAE_N_EPOCHS="$sae_epochs" SAE_OUTPUT="$checkpoint" SAE_RECORD="$records" "$julia_bin" --project=scripts scripts/pendulum/train_sae.jl || exit $?
    done
fi

if contains_stage retraction; then
    benchmark="$retraction_repo/scripts/retraction_accuracy.jl"
    [[ -f "$benchmark" ]] || { echo "missing retraction benchmark: $benchmark" >&2; exit 1; }
    run_stage retraction "$julia_bin" --project="$retraction_repo" "$benchmark" || exit $?
fi

echo "all requested stages completed"
