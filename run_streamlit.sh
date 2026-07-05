#!/bin/bash
set -e

export PYTHONPATH=src
streamlit run src/streamlit_app/app.py --server.port=8501 --server.address=localhost
