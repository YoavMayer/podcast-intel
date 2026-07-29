#!/usr/bin/env bash
# =============================================================================
# check-no-leak.sh -- data-leakage gate for the published tree
# =============================================================================
#
# podcast-intel is a public, MIT-licensed framework. It must never publish:
#   * strings identifying the private show it was extracted from,
#   * real people's names from that show's roster,
#   * episode audio, transcripts, briefs or databases,
#   * API keys or bearer tokens.
#
# The gate scans ONLY tracked files (`git ls-files`) -- that is exactly what a
# clone or a fork receives. Untracked local working data is not this script's
# business; .gitignore is.
#
# Usage:
#   scripts/check-no-leak.sh            # scan the whole tracked tree
#   scripts/check-no-leak.sh f1 f2 ...  # scan just these files (pre-commit)
#
# Exit codes: 0 = clean, 1 = at least one finding.
#
# Optional: if $PODCAST_SPURS points at a local private-show checkout, the
# script additionally reads its entities.yaml / speakers.yaml roster and greps
# for those names too. The private list is NEVER vendored into this repo; in CI
# the literal fallback list below is what runs.
# =============================================================================

set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

FINDINGS=0

report() {
    FINDINGS=$((FINDINGS + 1))
    printf 'LEAK: %s\n' "$1"
}

# -----------------------------------------------------------------------------
# Files in scope
# -----------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
    FILES=("$@")
else
    mapfile -t FILES < <(git ls-files)
fi

# This script necessarily contains the patterns it searches for.
# Sample transcript output is deliberate published example data.
SELF="scripts/check-no-leak.sh"
ALLOWLIST=("$SELF" "examples/sample-output/transcript_snippet.json")

in_allowlist() {
    local f
    for f in "${ALLOWLIST[@]}"; do
        [ "$1" = "$f" ] && return 0
    done
    return 1
}

SCAN=()
for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    in_allowlist "$f" && continue
    SCAN+=("$f")
done

# -----------------------------------------------------------------------------
# 1. Content patterns
# -----------------------------------------------------------------------------
# Case-insensitive; -I skips binaries.
CI_PATTERNS=(
    'tottenham|hotspur|\bspurs\b|\bthfc\b|\bcoys\b'
    'spurs[_-]israel'
    '#132257'
    'Rafi Ben David|Hanan Stein|Yoav Meir|Maor Elbaz'
    'Son Heung-min|Ange Postecoglou|Daniel Levy|James Maddison|Cristian Romero'
)

# Case-sensitive (Hebrew has no case; secrets and ids are case-bearing).
CS_PATTERNS=(
    'ספרס|טוטנהאם|הוטספר'
    'רפי בן דוד|חנן שטיין|יואב מאיר|מאור אלבאס'
    # Real Anchor feed ids are >=8 hex chars; the documented "abc123" placeholder
    # is 6 and stays legal on purpose.
    'anchor\.fm/s/[0-9a-f]{8,}'
    # Episode GUIDs
    '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    # Credentials
    'sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}'
)

if [ "${#SCAN[@]}" -gt 0 ]; then
    for pat in "${CI_PATTERNS[@]}"; do
        while IFS= read -r hit; do
            [ -n "$hit" ] && report "show/roster string -- $hit"
        done < <(grep -I -n -i -E "$pat" -- "${SCAN[@]}" 2>/dev/null | sed 's/^/  /')
    done
    for pat in "${CS_PATTERNS[@]}"; do
        while IFS= read -r hit; do
            [ -n "$hit" ] && report "private identifier/secret -- $hit"
        done < <(grep -I -n -E "$pat" -- "${SCAN[@]}" 2>/dev/null | sed 's/^/  /')
    done

    # Optional private roster, read from a local checkout only. Never vendored.
    if [ -n "${PODCAST_SPURS:-}" ]; then
        for roster in "$PODCAST_SPURS"/specializations/*/entities.yaml \
                      "$PODCAST_SPURS"/specializations/*/speakers.yaml; do
            [ -r "$roster" ] || continue
            while IFS= read -r name; do
                [ "${#name}" -ge 5 ] || continue
                while IFS= read -r hit; do
                    [ -n "$hit" ] && report "private roster name -- $hit"
                done < <(grep -I -n -F -- "$name" "${SCAN[@]}" 2>/dev/null | sed 's/^/  /')
            done < <(grep -oE '"[^"]{5,}"|- [^#]+' "$roster" 2>/dev/null |
                     sed 's/^- //; s/"//g; s/[[:space:]]*$//' | sort -u)
        done
    fi
fi

# -----------------------------------------------------------------------------
# 2. Forbidden directories -- must never exist as tracked paths
# -----------------------------------------------------------------------------
FORBIDDEN_DIRS='^(data|audio|reports|transcripts|briefs|briefings|specializations|logs|\.a5c)/'
while IFS= read -r f; do
    [ -n "$f" ] && report "forbidden directory -- $f"
done < <(printf '%s\n' "${FILES[@]}" | grep -E "$FORBIDDEN_DIRS")

# -----------------------------------------------------------------------------
# 3. Forbidden file shapes
# -----------------------------------------------------------------------------
FORBIDDEN_SHAPES='(\.(mp3|wav|m4a|flac|srt|vtt|db|sqlite|sqlite3)$)|((^|/)transcript[^/]*\.json$)|((^|/)(content_metrics|pqs_recomputed|episodes)\.json$)|((^|/)\.env)'
while IFS= read -r f; do
    [ -n "$f" ] || continue
    in_allowlist "$f" && continue
    report "forbidden file shape -- $f"
done < <(printf '%s\n' "${FILES[@]}" | grep -E "$FORBIDDEN_SHAPES")

# -----------------------------------------------------------------------------
if [ "$FINDINGS" -gt 0 ]; then
    printf '\ncheck-no-leak: %d finding(s). Nothing above may be published.\n' "$FINDINGS" >&2
    exit 1
fi

printf 'check-no-leak: clean (%d tracked files scanned).\n' "${#SCAN[@]}"
exit 0
