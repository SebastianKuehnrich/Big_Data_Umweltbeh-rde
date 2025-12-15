#!/bin/bash
# Startet das EPA Air Quality Dashboard
echo "Starte EPA Air Quality Dashboard..."
echo ""

if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtuelle Umgebung nicht gefunden!"
    echo "Bitte führen Sie zuerst ./setup.sh aus."
    exit 1
fi

source .venv/bin/activate
streamlit run epa_dashboard2.py

