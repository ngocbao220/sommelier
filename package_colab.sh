#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="${PACKAGE_NAME:-sommelier_refactor_colab}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${SCRIPT_DIR}/${PACKAGE_NAME}.zip}"
TMP_ROOT="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_ROOT}"
}
trap cleanup EXIT

mkdir -p "${TMP_ROOT}/${PACKAGE_NAME}"
rsync -a \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude "outputs" \
  --exclude "${PACKAGE_NAME}.zip" \
  "${SCRIPT_DIR}/" "${TMP_ROOT}/${PACKAGE_NAME}/"

(
  cd "${TMP_ROOT}"
  zip -qr "${ARCHIVE_PATH}" "${PACKAGE_NAME}"
)

echo "Created ${ARCHIVE_PATH}"
