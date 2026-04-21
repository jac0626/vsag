#!/usr/bin/env bash

get_os() {
    local kernel
    kernel="$(uname -s)"

    case "${kernel}" in
        Darwin)
            local product_name product_version build_version
            product_name="$(sw_vers -productName)"
            product_version="$(sw_vers -productVersion)"
            build_version="$(sw_vers -buildVersion)"
            echo "- OS: ${product_name} ${product_version} (${build_version})"
            ;;
        Linux)
            if [[ -r /etc/os-release ]]; then
                # shellcheck disable=SC1091
                . /etc/os-release
                echo "- OS: ${PRETTY_NAME:-${NAME:-Linux}}"
            else
                echo "- OS: Linux $(uname -r)"
            fi
            ;;
        *)
            echo "- OS: ${kernel} $(uname -r)"
            ;;
    esac

    echo "- arch: $(uname -m)"
}

get_macos_details() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        return 0
    fi

    local xcode_path
    if xcode_path="$(xcode-select -p 2>/dev/null)"; then
        echo "- xcode-select: ${xcode_path}"
    fi

    if command -v pkgutil >/dev/null 2>&1; then
        local clt_version
        clt_version="$(pkgutil --pkg-info=com.apple.pkg.CLTools_Executables 2>/dev/null | awk -F': ' '/version/ {print $2; exit}')"
        if [[ -n "${clt_version}" ]]; then
            echo "- Xcode CLT version: ${clt_version}"
        fi
    fi

    if command -v brew >/dev/null 2>&1; then
        echo "- Homebrew: $(brew --version | head -n 1)"
        echo "- Homebrew prefix: $(brew --prefix)"
    fi
}

get_vsag() {
    local vsag_version
    if vsag_version="$(git describe --tags --always --dirty --match "v*" 2>/dev/null)"; then
        echo "- vsag version: ${vsag_version}"
    else
        echo "- vsag version: unavailable"
    fi
}

get_compiler() {
    local compiler_version
    if compiler_version="$(${CXX:-c++} --version 2>/dev/null | head -n 1)"; then
        echo "- compiler version: ${compiler_version}"
    else
        echo "- compiler version: unavailable"
    fi
}

get_os
get_macos_details
get_vsag
get_compiler
