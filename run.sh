#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 download_model.py
  exec python3 app.py
fi
python download_model.py
exec python app.py
