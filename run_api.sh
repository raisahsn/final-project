#!/bin/bash
set -e

export PYTHONPATH=src
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
