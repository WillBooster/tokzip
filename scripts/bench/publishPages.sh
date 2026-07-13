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

# Scoped to this process only (not `git config`), so local runs never mutate .git/config.
export GIT_AUTHOR_NAME='github-actions[bot]' GIT_COMMITTER_NAME='github-actions[bot]'
export GIT_AUTHOR_EMAIL='41898282+github-actions[bot]@users.noreply.github.com'
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

rm -rf "$pages_dir"
git worktree prune
if git ls-remote --exit-code --heads origin gh-pages > /dev/null; then
  # Base the branch on FETCH_HEAD instead of origin/gh-pages: actions/checkout clones
  # single-branch, so the remote-tracking ref for gh-pages is never created (and a
  # fetch.prune=true config can even delete it mid-fetch).
  git fetch origin gh-pages
  git worktree add "$pages_dir" -B gh-pages FETCH_HEAD
else
  # First publish: create an orphan gh-pages branch inside a detached worktree. A stale
  # local gh-pages branch (e.g. from an aborted run) would make --orphan fail, so drop it.
  git branch -D gh-pages 2> /dev/null || true
  git worktree add --detach "$pages_dir"
  git -C "$pages_dir" checkout --orphan gh-pages
  git -C "$pages_dir" rm -rfq . 2> /dev/null || true
fi

cp scripts/bench/site/index.html "$pages_dir/index.html"
# Static SVG figures for the README, regenerated from this run so they track main.
# Clear the directory first: a chart the renderer no longer emits (renamed, or skipped
# for a report without speed data) must not keep shipping as if it belonged to this run.
rm -rf "$pages_dir/charts"
bun scripts/bench/renderCharts.ts "$result_json" "$pages_dir/charts"
touch "$pages_dir/.nojekyll" "$pages_dir/results.jsonl"
grep -v "\"commit\":\"$commit\"" "$pages_dir/results.jsonl" > "$pages_dir/results.jsonl.tmp" || true
jq -c . "$result_json" >> "$pages_dir/results.jsonl.tmp"
mv "$pages_dir/results.jsonl.tmp" "$pages_dir/results.jsonl"

git -C "$pages_dir" add .nojekyll index.html results.jsonl charts
if git -C "$pages_dir" diff --cached --quiet; then
  echo 'no changes to publish'
else
  git -C "$pages_dir" commit -m "bench: ${commit:0:12}"
  git -C "$pages_dir" push origin gh-pages
fi
git worktree remove --force "$pages_dir"
