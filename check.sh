#!/usr/bin/env bash
# check.sh — Validate prerequisites before deploying.
#
# Override the config file via CONFIG_PATH env var (default: config.toml).
# Exits 0 if all required checks pass (warnings are non-fatal).
# Exits 1 if any required check fails.
set -uo pipefail

export CONFIG_PATH="${CONFIG_PATH:-config.toml}"

all_ok=true

pass()    { echo "  [OK]   $1"; }
fail()    { echo "  [FAIL] $1${2:+  →  $2}"; all_ok=false; }
warn()    { echo "  [WARN] $1${2:+  →  $2}"; }
has_cmd() { command -v "$1" &>/dev/null; }

echo ""
echo "── Required tools ───────────────────────────────────────────"
for tool in uv docker gcloud kubectl envsubst; do
    if has_cmd "$tool"; then
        pass "$tool"
    else
        fail "$tool" "install $tool and ensure it is on PATH"
    fi
done

echo ""
echo "── GCP credentials ──────────────────────────────────────────"
adc="$HOME/.config/gcloud/application_default_credentials.json"
if [[ -f "$adc" ]]; then
    pass "application-default credentials"
else
    fail "application-default credentials ($adc)" "run: gcloud auth application-default login"
fi

echo ""
echo "── CloudVolume secrets ──────────────────────────────────────"
secrets_dir="$HOME/.cloudvolume/secrets"
for secret in google-secret.json cave-secret.json; do
    if [[ -f "$secrets_dir/$secret" ]]; then
        pass "$secrets_dir/$secret"
    else
        fail "$secrets_dir/$secret" "add $secret to $secrets_dir/"
    fi
done

echo ""
echo "── config.toml placeholders ─────────────────────────────────"
if [[ -f "$CONFIG_PATH" ]]; then
    for ph in "your-dockerhub-user" "my-gcp-project" "my-bucket"; do
        if grep -q "$ph" "$CONFIG_PATH"; then
            fail "placeholder \"$ph\" still in $CONFIG_PATH" "replace with your actual value"
        else
            pass "no placeholder \"$ph\""
        fi
    done
else
    fail "$CONFIG_PATH exists" "create $CONFIG_PATH or set CONFIG_PATH to the right file"
fi

echo ""
echo "── Optional tools ───────────────────────────────────────────"
if has_cmd kind; then
    pass "kind (for --local testing)"
else
    warn "kind (for --local testing)" "brew install kind"
fi

echo ""
if [[ "$all_ok" == true ]]; then
    echo "All required checks passed."
    exit 0
else
    echo "One or more required checks failed. Fix the issues above before deploying."
    exit 1
fi
