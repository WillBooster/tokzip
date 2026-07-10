#!/usr/bin/env bash
# Publishes one benchmark result to the `gh-pages` branch: appends the JSON report as a
# line of `results.jsonl` (replacing any previous entry for the same commit, so re-runs
# stay idempotent) and refreshes the static dashboard copied from `scripts/bench/site/`.
#
# Usage: bash scripts/bench/publishPages.sh <bench.json>
set -euo pipefail

result_json="${1:?usage: publishPages.sh <bench.json>}"
commit="$(jq -r .commit "$result_json")"
pages_dir=.tmp/gh-pages

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'

rm -rf "$pages_dir"
git worktree prune
if git ls-remote --exit-code --heads origin gh-pages > /dev/null; then
  git fetch origin gh-pages
  git worktree add "$pages_dir" -B gh-pages origin/gh-pages
else
  # First publish: create an orphan gh-pages branch inside a detached worktree.
  git worktree add --detach "$pages_dir"
  git -C "$pages_dir" checkout --orphan gh-pages
  git -C "$pages_dir" rm -rfq . 2> /dev/null || true
fi

cp scripts/bench/site/index.html "$pages_dir/index.html"
touch "$pages_dir/.nojekyll" "$pages_dir/results.jsonl"
grep -v "\"commit\":\"$commit\"" "$pages_dir/results.jsonl" > "$pages_dir/results.jsonl.tmp" || true
jq -c . "$result_json" >> "$pages_dir/results.jsonl.tmp"
mv "$pages_dir/results.jsonl.tmp" "$pages_dir/results.jsonl"

git -C "$pages_dir" add .nojekyll index.html results.jsonl
if git -C "$pages_dir" diff --cached --quiet; then
  echo 'no changes to publish'
else
  git -C "$pages_dir" commit -m "bench: ${commit:0:12}"
  git -C "$pages_dir" push origin gh-pages
fi
git worktree remove --force "$pages_dir"
