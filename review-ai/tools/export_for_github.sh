#!/usr/bin/env bash
# Build (or refresh) the fresh export that is pushed to the private GitHub copy.
#
# The working repository must never be pushed: its history before 39e3c20 contains licensed files
# (Rakuten Travel data, NII-IDR agreement, article 3(5): no provision to third parties).
# The export has no history of the working repository and leaves out:
#   review-ai/legacy/                               earlier Mercari project (licensed comments in its past data)
#   review-ai/data/panel_exclude_text_hashes.jsonl  hashes of licensed review texts (derived data)
# Everything else is the committed tree of the working repository, which tools/verify_repo.py checks
# for review text before every export.
#
# Each run adds one commit "Export of <sha>" to github_export/<name>, so later pushes need no force.
# GitHub login stays inside gujincheng/.gh_config (the server account is shared); git uses it only
# through this export's own repository config, never the global credential store.
#
# Usage:  bash review-ai/tools/export_for_github.sh
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
NAME=rakuten-absa-weak-supervision
OUT="$ROOT/github_export/$NAME"
GH_DIR="$ROOT/.gh_config"
EXCLUDE=(review-ai/legacy review-ai/data/panel_exclude_text_hashes.jsonl)

[[ -z "$(git status --porcelain)" ]] || { echo "commit local changes first"; exit 1; }
echo "[check] licensed-text guard and the other self-checks"
review-ai/.venv/bin/python review-ai/tools/verify_repo.py > /dev/null || { echo "verify_repo failed"; exit 1; }

mkdir -p "$OUT"
if [[ ! -d "$OUT/.git" ]]; then
  git init -q -b main "$OUT"
  git -C "$OUT" config user.name "$(git config user.name)"
  git -C "$OUT" config user.email "$(git config user.email)"
  git -C "$OUT" config --replace-all credential.helper ""  # drop the shared global helpers
  git -C "$OUT" config --add credential.helper "!GH_CONFIG_DIR=$GH_DIR $(command -v gh) auth git-credential"
fi
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git archive HEAD | tar -x -C "$OUT"
for p in "${EXCLUDE[@]}"; do rm -rf "${OUT:?}/${p:?}"; done

SRC="$(git rev-parse --short HEAD)"
cat > "$OUT/EXPORT.md" <<EOF
# About this repository

This is a fresh export of the working repository at commit \`$SRC\`, without its history.
The working repository is never pushed, because its early history contains licensed data.

Left out of the export:
- \`review-ai/legacy/\`: an earlier, unrelated project (Mercari comments, 3-class sentiment).
- \`review-ai/data/panel_exclude_text_hashes.jsonl\`: hashes of licensed review texts used to remove
  Gold sentences from the pretraining pool; rebuilt from the licensed data by \`tools/check_panel_gold_overlap.py\`
  and \`tools/approx_dedup_panel_gold.py\` (REPRODUCE.md, section on the exclusion list).

No Rakuten Travel data, Gold corpus text or derived per-sentence files are included; see README.md
("Data") for how to obtain the data under the NII-IDR agreement.
EOF

big=$(find "$OUT" -path "$OUT/.git" -prune -o -type f -size +50M -print)
[[ -z "$big" ]] || { echo "files over 50 MB: $big"; exit 1; }
git -C "$OUT" add -A
if git -C "$OUT" diff --cached --quiet; then
  echo "export unchanged"
else
  git -C "$OUT" commit -q -m "Export of working repository at $SRC"
  echo "export committed: $(git -C "$OUT" log --oneline -1) ($(git -C "$OUT" ls-files | wc -l) files)"
fi
