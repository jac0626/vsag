#!/usr/bin/env bash

set -euo pipefail

ensure_homebrew() {
    if command -v brew >/dev/null 2>&1; then
        return 0
    fi

    for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        if [[ -x "${candidate}" ]]; then
            eval "$("${candidate}" shellenv)"
            return 0
        fi
    done

    echo "Homebrew is required on macOS. Install it from https://brew.sh/ and re-run this script." >&2
    exit 1
}

ensure_xcode_clt() {
    if xcode-select -p >/dev/null 2>&1; then
        return 0
    fi

    echo "Xcode Command Line Tools are required on macOS. Run 'xcode-select --install' first." >&2
    exit 1
}

main() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "This script is intended for macOS only." >&2
        exit 1
    fi

    ensure_xcode_clt
    ensure_homebrew

    local formulae=(
        ccache
        cmake
        gcc
        libomp
        openblas
        python
    )

    brew install "${formulae[@]}"
}

main "$@"
