#!/usr/bin/env bash
# browser_up.sh — idempotent warm-up for the jev-ultrafast browser stack.
# Fixes the two recurring failures:
#   1) stale SingletonLock (chromium killed, lock left pointing at a dead pid)
#   2) stale browser-harness fatal log (ensure_daemon reads it and refuses to start)
# Usage: browser_agent/browser_up.sh
set -e
pkill -f "remote-debugging-port=9222" 2>/dev/null || true
pkill -f "browser_harness.daemon" 2>/dev/null || true
sleep 2
rm -f "$HOME/.config/chromium/SingletonLock" "$HOME/.config/chromium/SingletonCookie" "$HOME/.config/chromium/SingletonSocket"
rm -f "$HOME/.config/browser-harness/tmp/bu-default.log"
nohup /usr/bin/chromium --headless=new --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.config/chromium" --no-first-run --no-default-browser-check \
  --disable-gpu about:blank > /tmp/chromium_cdp.log 2>&1 &
for i in $(seq 1 15); do
  curl -s -m 2 http://127.0.0.1:9222/json/version >/dev/null && { echo "CDP ready"; exit 0; }
  sleep 1
done
echo "CDP failed to come up"; exit 1
