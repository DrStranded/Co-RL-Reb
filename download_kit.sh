#!/usr/bin/env bash
# Fetch everything this folder does not contain itself, from public sources, and verify it:
#   code/<commit>/  the original training code, taken BY COMMIT SHA from the public trl-projects repository
#                   (git checks every object against its hash, so the code is exactly that commit)
#   wheelhouse_<env>/, extra/  Python packages, Python itself and flash-attn, from PyPI, download.pytorch.org and
#                   GitHub releases; every file's sha256 must equal the copy that was validated, or the script stops.
# Safe to rerun: anything already present and correct is skipped.
#   bash download_kit.sh collab      # needed for ttrl_llama3b and corlplus_qrephr3b (about 5 GB)
#   bash download_kit.sh anvil       # only needed for ttrl_qwen3b (about 4.6 GB)
# Needs git, curl, sha256sum and access to github.com, pypi.org, files.pythonhosted.org, download.pytorch.org.
set -euo pipefail
ENVNAME="${1:?usage: download_kit.sh collab|anvil}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/williamium3000/trl-projects.git}"
case "$ENVNAME" in
  collab) COMMITS="05a21d7fb49e0829753b4dff1057ab872b8cb3d8 00e0eed553dd69cae24ef5b13a251a65ab1b122b" ;;
  anvil)  COMMITS="4dc11923c53957388e5ed77a4e3fec587a393475" ;;
  *) echo "unknown env $ENVNAME"; exit 2 ;;
esac
for t in git curl sha256sum tar; do command -v $t >/dev/null || { echo "FATAL: need $t"; exit 1; }; done
# 1) code, by commit SHA
for c in $COMMITS; do
  d="$KIT/code/${c:0:8}"
  if [ -f "$d/ORIGINAL_COMMIT" ] && [ "$(cat "$d/ORIGINAL_COMMIT")" = "$c" ]; then continue; fi
  tmp=$(mktemp -d); rm -rf "$d"; mkdir -p "$d"
  git -C "$tmp" init -q && git -C "$tmp" fetch -q --depth 1 "$REPO_URL" "$c"
  [ "$(git -C "$tmp" rev-parse FETCH_HEAD)" = "$c" ] || { echo "FATAL: fetched commit is not $c"; exit 1; }
  git -C "$tmp" archive "$c" | tar -x -C "$d" && echo "$c" > "$d/ORIGINAL_COMMIT"; rm -rf "$tmp"
  echo "  code/${c:0:8} <- $REPO_URL @ $c"
done
# 2) packages, by sha256
LIST="$KIT/kit_files_$ENVNAME.tsv"; [ -f "$LIST" ] || { echo "FATAL: $LIST missing"; exit 1; }
total=$(grep -c . "$LIST"); n=0; fail=0
while IFS=$'\t' read -r rel sha url; do
  n=$((n+1)); dst="$KIT/$rel"; mkdir -p "$(dirname "$dst")"
  if [ -f "$dst" ] && [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$sha" ]; then continue; fi
  ok=0
  for attempt in 1 2 3 4 5; do
    if curl -fsSL --retry 3 --connect-timeout 30 -o "$dst.part" "$url" </dev/null \
       && [ "$(sha256sum "$dst.part" | cut -d' ' -f1)" = "$sha" ]; then mv "$dst.part" "$dst"; ok=1; break; fi
    sleep $((attempt * 5))
  done
  if [ "$ok" = 1 ]; then [ $((n % 25)) = 0 ] && echo "  $n/$total"
  else echo "FAILED ($n/$total): $rel <- $url"; rm -f "$dst.part"; fail=$((fail + 1)); fi
done < "$LIST"
[ "$fail" = 0 ] || { echo "FATAL: $fail file(s) could not be downloaded with the expected sha256"; exit 1; }
echo "ALL CODE AND $total PACKAGE FILES PRESENT AND VERIFIED -> next: setup_env_arm64.sh $ENVNAME"
