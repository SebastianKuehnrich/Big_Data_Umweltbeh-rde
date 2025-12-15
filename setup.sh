#!/bin/bash
# Setup-Script für EPA Air Quality Dashboard
# macOS/Linux Version

echo "================================"
echo "EPA Air Quality Dashboard Setup"
echo "================================"
echo ""

# Prüfen ob Python installiert ist
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 ist nicht installiert!"
    echo "Bitte installieren Sie Python 3.11 oder höher"
    exit 1
fi

echo "[1/5] Python gefunden!"
python3 --version

# Virtuelle Umgebung erstellen
echo ""
echo "[2/5] Erstelle virtuelle Umgebung..."
if [ -d ".venv" ]; then
    echo "Virtuelle Umgebung existiert bereits."
else
    python3 -m venv .venv
    echo "Virtuelle Umgebung erstellt!"
fi

# Virtuelle Umgebung aktivieren
echo ""
echo "[3/5] Aktiviere virtuelle Umgebung..."
source .venv/bin/activate

# Dependencies installieren
echo ""
echo "[4/5] Installiere Dependencies..."
echo "Dies kann einige Minuten dauern..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Installation fehlgeschlagen!"
    exit 1
fi

echo ""
echo "[5/5] Prüfe Datenverzeichnis..."
if [ ! -d "Data" ]; then
    mkdir Data
    echo "Data-Ordner erstellt."
    echo "HINWEIS: Bitte legen Sie die CSV-Datei in den Data-Ordner!"
else
    echo "Data-Ordner existiert."
fi

echo ""
echo "================================"
echo "Setup erfolgreich abgeschlossen!"
echo "================================"
echo ""
echo "Um das Dashboard zu starten:"
echo "  1. Aktiviere die virtuelle Umgebung: source .venv/bin/activate"
echo "  2. Starte Streamlit: streamlit run epa_dashboard2.py"
echo ""
echo "Oder führen Sie ./start_dashboard.sh aus"
echo ""

