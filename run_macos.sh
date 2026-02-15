#!/bin/bash
set -e
cd "$(dirname "$0")"
[[ -d venv ]] || python3 -m venv venv
echo "Installing dependencies..."
venv/bin/pip install --trusted-host pypi.org --trusted-host files.pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
echo "Starting app..."
exec venv/bin/python app.py