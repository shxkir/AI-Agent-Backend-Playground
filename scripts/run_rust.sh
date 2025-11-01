#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}/rust_api"

echo "Starting Rust API (Actix) service..."
cargo run
