#!/bin/bash

sudo apt install python3

sudo apt install pip

PYTHON_COMMAND="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_COMMAND="python"
fi

echo "Installing required libraries."
$PYTHON_COMMAND -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
  echo "Error installing libraries."
  exit 1
fi

echo "Running project."
$PYTHON_COMMAND main.py

exit 0
