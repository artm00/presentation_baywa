#!/bin/bash
# Double-click this file on a Mac to start the app.
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Please install it from https://www.python.org/downloads/ and try again."
  read -n 1 -s -r -p "Press any key to close."; exit 1
fi
if [ ! -d ".venv" ]; then
  echo "First start: setting up (takes 1-3 minutes, only once)..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip -q
  ./.venv/bin/python -m pip install -r requirements.txt -q
fi
echo "Starting the app - your browser will open. Keep this window open while you use it."
./.venv/bin/python -m streamlit run app.py
