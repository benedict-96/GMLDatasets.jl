#!/usr/bin/env bash
#
# Run `scripts/geometric_optimizers/mnist_cuda_repetitions.jl` on the RTX 4090 workstation: the same configuration
# several times, reported as a mean and a sample standard deviation.
#
#   screen -S mnist                                     # then, inside it:
#   scripts/geometric_optimizers/run_mnist_repetitions.sh --smoke            # 3 × 2 epochs, ≈4 min
#   scripts/geometric_optimizers/run_mnist_repetitions.sh                    # 10 × 500 epochs of Adam on Stiefel
#   scripts/geometric_optimizers/run_mnist_repetitions.sh --repeat 3         # 3 of them, ≈4:45 h
#   scripts/geometric_optimizers/run_mnist_repetitions.sh -c all --repeat 3   # all four configurations, ≈21 h
#
# Detach with `C-a d`, log out, come back with `screen -r mnist`. Or start it detached in one
# go, which is what an ssh session is for:
#
#   screen -dmS mnist scripts/geometric_optimizers/run_mnist_repetitions.sh
#
# One repetition is one configuration of the full run, i.e. ≈1:35 h for `Adam` on an RTX 4090 —
# multiply. This is the sibling of `run_mnist_cuda.sh`, which runs all four configurations
# *once* and is what the learning curves of the documentation come from; this one answers how
# far a single accuracy from it can be trusted. It leaves five files behind in
# `results/`:
#
#   <stamp>_report.txt        the environment, the gradient checks, one line per epoch and
#                             repetition, the statistics and the verdict, flushed as it goes
#   <stamp>_losses.csv        one row per optimizer step:
#                             run, configuration, repetition, epoch, batch, step, loss
#   <stamp>_parameters.jld2   the parameters, losses, timings and accuracies of every
#                             repetition, plus the samples the statistics come from
#   <stamp>_runs.csv          one machine-readable outcome/timing record per repetition
#   <stamp>_stdout.txt        everything the process wrote, including a backtrace if it died
#
# `<stamp>` is the start time, so a second run never overwrites the first. Copy all four off
# the machine when it is done — they are self-contained.
#
# Run this from anywhere; it changes to the repository root itself.

set -euo pipefail

cd "$(dirname "$0")/.."

julia_bin="${JULIA:-julia}"
mode="repetitions"
repetitions=""      # what `--repeat` said, if it was given at all
configurations="${MNIST_CONFIGURATIONS:-geometric-adam-cayley}"

usage() {
    echo "usage: $0 [--smoke] [--repeat N] [-c|--configurations LIST]" >&2
    echo "  LIST is comma separated: geometric-adam-cayley, standard-adam, gradient, momentum, or all" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) mode="smoke"; shift ;;
        --repeat) repetitions="${2:-}"; shift 2 ;;
        -c|--configurations) configurations="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# `--repeat` wins over `MNIST_REPETITIONS`, which wins over the default — and the default of a
# smoke test is three rather than ten, so that it stays a smoke test when neither was given
# while still computing a deviation over more than two samples.
if [ -z "$repetitions" ]; then
    if [ "$mode" = "smoke" ]; then
        repetitions="${MNIST_REPETITIONS:-3}"
    else
        repetitions="${MNIST_REPETITIONS:-10}"
    fi
fi

case "$repetitions" in
    ''|*[!0-9]*) echo "--repeat takes a positive integer, got '$repetitions'" >&2; exit 2 ;;
esac
[ "$repetitions" -ge 1 ] || { echo "--repeat takes a positive integer" >&2; exit 2; }
[ -n "$configurations" ] || { echo "--configurations takes a non-empty list" >&2; exit 2; }

stamp="$(date +%Y%m%d_%H%M%S)_${mode}"
mkdir -p results

export MNIST_REPORT="results/${stamp}_report.txt"
export MNIST_LOSSES="results/${stamp}_losses.csv"
export MNIST_OUTPUT="results/${stamp}_parameters.jld2"
export MNIST_RECORDS="results/${stamp}_runs.csv"
export MNIST_REPETITIONS="$repetitions"
export MNIST_CONFIGURATIONS="$configurations"
stdout_log="results/${stamp}_stdout.txt"

if [ "$mode" = "smoke" ]; then
    # Two epochs per repetition, evaluated every epoch: enough to see the gradient check pass,
    # every repetition execute, the statistics be computed over more than one sample and all
    # three files be written — and its epoch lines are what the schedule of the real run should
    # be extrapolated from.
    export MNIST_N_EPOCHS=2
    export MNIST_ACCURACY_EVERY=1
fi

# The progress line is a carriage return per batch. It is what tells you the first repetition is
# moving, so it goes to the terminal, but not into the log that `tee` writes.
export MNIST_PROGRESS=0

echo "mode              $mode"
echo "julia             $("$julia_bin" --version)"
echo "repetitions       $MNIST_REPETITIONS"
echo "configurations    $MNIST_CONFIGURATIONS"
echo "seeds             ${MNIST_VARY_SEED:-1} (0 = the same seed every time)"
echo "report            $MNIST_REPORT"
echo "loss curves       $MNIST_LOSSES"
echo "parameters        $MNIST_OUTPUT"
echo "run records       $MNIST_RECORDS"
echo "stdout            $stdout_log"
echo "follow it with    tail -f $MNIST_REPORT"
echo

# A machine that has not run this before has no `scripts/Manifest.toml`. Resolving it here
# means a dependency that cannot be installed says so now, in one line, rather than as a
# precompilation error out of the script itself — and `errexit` stops before MNIST is even
# downloaded. It is a no-op once the environment exists.
"$julia_bin" --project=scripts --startup-file=no -e 'using Pkg; Pkg.instantiate()'
echo

# `--startup-file=no` so that nothing in the user's `startup.jl` reaches a run of this length.
# `stdbuf` keeps the redirected output line buffered, so `tail -f` on it is worth something;
# it is not on every machine, hence the fallback.
> "$stdout_log"

# `set +e` around the pipeline: with `errexit` and `pipefail` a julia that dies would take this
# script down with it before it could say where it left its output, which is the one thing it
# has to get right.
set +e
if command -v stdbuf > /dev/null; then
    stdbuf -oL -eL "$julia_bin" --project=scripts --startup-file=no scripts/geometric_optimizers/mnist_cuda_repetitions.jl 2>&1 | tee "$stdout_log"
else
    "$julia_bin" --project=scripts --startup-file=no scripts/geometric_optimizers/mnist_cuda_repetitions.jl 2>&1 | tee "$stdout_log"
fi
status="${PIPESTATUS[0]}"
set -e

echo
if [ "$status" -eq 0 ]; then
    echo "done. The statistics are at the end of the report:"
    echo "    tail -40 $MNIST_REPORT"
    echo "copy these off the machine:"
    echo "    $MNIST_REPORT"
    echo "    $MNIST_LOSSES"
    echo "    $MNIST_OUTPUT"
    echo "    $MNIST_RECORDS"
    echo "    $stdout_log"
    echo "e.g. from the other machine:"
    echo "    scp '<user>@<host>:$(pwd)/results/${stamp}_*' ."
else
    echo "julia exited with $status — the report is complete up to where it stopped:"
    echo "    $MNIST_REPORT"
    echo "and the backtrace is at the end of:"
    echo "    $stdout_log"
fi

exit "$status"
