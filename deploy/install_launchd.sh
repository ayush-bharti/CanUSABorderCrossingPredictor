#!/bin/sh
# Install the launchd jobs that log forecasts hourly and refresh the models monthly, for the current user.
# Run again after moving the checkout; "uninstall" removes them. launchd runs a job missed while the Mac
# slept when it wakes, but not one missed while it was shut down.
set -eu
repo=$(cd "$(dirname "$0")/.." && pwd)
agents="$HOME/Library/LaunchAgents"
logs="$HOME/Library/Logs/border-predictor"
for job in track refresh; do
    label="com.borderpredictor.$job"
    plist="$agents/$label.plist"
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    rm -f "$plist"
    if [ "${1:-}" = uninstall ]; then
        echo "Removed $label"
        continue
    fi
    mkdir -p "$agents" "$logs"
    sed -e "s|__REPO__|$repo|g" -e "s|__PYTHON__|$repo/venv/bin/python|g" -e "s|__LOGS__|$logs|g" \
        "$repo/deploy/$label.plist" > "$plist"
    launchctl bootstrap "gui/$(id -u)" "$plist"
    echo "Installed $label, logging to $logs/$job.log"
done
