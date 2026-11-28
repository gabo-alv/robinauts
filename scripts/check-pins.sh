#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors
#
# neorc-from-source: checks the neorc, neorc-core and oneberth pins of
# backend/pyproject.toml in seconds. neorc and neorc-core pin one commit, each
# commit is on its upstream main, and every uv.lock holds the same commits.
# Every check runs, and each failure names the file and the commit.
set -eu

if [ "$#" -ne 0 ]; then
    printf '%s takes no arguments (got: %s)\n' "$0" "$*" >&2
    exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
pyproject="$root/backend/pyproject.toml"

# One field of a package's line in [tool.uv.sources]: git, rev or subdirectory.
source_field() {
    sed -n "s/^$1 = .*$2 = \"\([^\"]*\)\".*/\1/p" "$pyproject"
}

for package in neorc neorc-core oneberth; do
    if [ -z "$(source_field "$package" git)" ] || [ -z "$(source_field "$package" rev)" ]; then
        printf 'backend/pyproject.toml pins no commit of %s\n' "$package" >&2
        exit 1
    fi
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
trap 'rm -rf "$work"; exit 130' INT
trap 'rm -rf "$work"; exit 143' TERM

failed=no
fail() {
    printf '%s\n' "$*" >&2
    failed=yes
}

neorc_revision=$(source_field neorc rev)
neorc_core_revision=$(source_field neorc-core rev)
if [ "$neorc_core_revision" != "$neorc_revision" ]; then
    fail "backend/pyproject.toml pins neorc at $neorc_revision and neorc-core at $neorc_core_revision; both come from one neorc commit"
fi

for package in neorc oneberth; do
    repository=$(source_field "$package" git)
    revision=$(source_field "$package" rev)
    git clone --quiet --filter=tree:0 --no-checkout --single-branch --branch main \
        "$repository" "$work/$package"
    if ! git -C "$work/$package" merge-base --is-ancestor "$revision" origin/main 2>/dev/null; then
        fail "backend/pyproject.toml pins $package at $revision, which is not on the main branch of $repository"
    fi
done

for lock in "$root/backend/uv.lock" "$root"/extra-workers/*/uv.lock; do
    name=${lock#"$root"/}
    for package in neorc neorc-core oneberth; do
        revision=$(source_field "$package" rev)
        subdirectory=$(source_field "$package" subdirectory | sed 's|/|%2F|g')
        url="$(source_field "$package" git)?subdirectory=$subdirectory&rev="
        locked=$(grep -o "$url[0-9a-f#]*" "$lock" | sed 's|^.*&rev=||' | tr '#' '\n' |
            sort -u | paste -sd ' ' -)
        if [ -z "$locked" ]; then
            fail "$name holds no git source of $package; backend/pyproject.toml pins $revision"
        elif [ "$locked" != "$revision" ]; then
            fail "$name locks $package at $locked, and backend/pyproject.toml pins $revision"
        fi
    done
done

if [ "$failed" = yes ]; then
    exit 1
fi
printf 'neorc %s and oneberth %s: on main, and in every lock\n' \
    "$neorc_revision" "$(source_field oneberth rev)" >&2
