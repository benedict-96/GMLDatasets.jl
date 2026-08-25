#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
session_name="${GML_SCREEN_SESSION:-gml-revision}"
runner_args=()

usage() {
    cat <<USAGE
usage: $0 [--session NAME] --smoke|--full [run_experiments.sh options]

Starts scripts/revision/run_experiments.sh in a detached GNU screen session.
USAGE
}

while (( $# )); do
    case "$1" in
        --session) session_name="${2:?missing session name}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) runner_args+=("$1"); shift ;;
    esac
done

[[ "$session_name" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "screen session name may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 2
}
(( ${#runner_args[@]} > 0 )) || { usage >&2; exit 2; }
if [[ " ${runner_args[*]} " != *" --smoke "* && " ${runner_args[*]} " != *" --full "* ]]; then
    echo "pass --smoke or --full explicitly" >&2
    exit 2
fi
command -v screen >/dev/null 2>&1 || {
    echo "GNU screen is required; install it before launching the experiment" >&2
    exit 1
}
if screen -ls 2>/dev/null | grep -Fq ".$session_name"; then
    echo "screen session already exists: $session_name" >&2
    echo "attach with: screen -r $session_name" >&2
    exit 1
fi

printf -v repo_quoted '%q' "$repo_root"
printf -v runner_command '%q ' "$repo_root/scripts/revision/run_experiments.sh" "${runner_args[@]}"
screen -DmS "$session_name" bash -lc "cd $repo_quoted && exec $runner_command"

echo "started detached screen session: $session_name"
echo "attach: screen -r $session_name"
echo "detach: press Ctrl-A, then D"
echo "list:   screen -ls"
