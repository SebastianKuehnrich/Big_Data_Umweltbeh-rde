# Schnellstart-Guide

## 🚀 In 3 Schritten zum laufenden Dashboard

### Windows

```bash
# 1. Setup ausführen
setup.bat

# 2. Dashboard starten
start_dashboard.bat
```

### macOS/Linux

```bash
# 1. Ausführbar machen
chmod +x setup.sh start_dashboard.sh

# 2. Setup ausführen
./setup.sh

# 3. Dashboard starten
./start_dashboard.sh
```

## 📋 Git & GitHub

### Repository erstellen

```bash
# Git initialisieren
git init

# Alle Dateien hinzufügen
git add .

# Ersten Commit
git commit -m "Initial commit: EPA Air Quality Dashboard v2.0"

# GitHub-Repo verbinden
git remote add origin https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git

# Branch umbenennen
git branch -M main

# Pushen
git push -u origin main
```

## 🚂 Railway Deployment

### Über Browser (Empfohlen)

1. Gehe zu [railway.app](https://railway.app)
2. Melde dich mit GitHub an
3. Klicke auf "New Project"
4. Wähle "Deploy from GitHub repo"
5. Wähle dein Repository
6. Warte 2-3 Minuten
7. Klicke auf "Generate Domain"
8. ✅ Fertig!

### Über CLI

```bash
# Railway CLI installieren
npm i -g @railway/cli

# Login
railway login

# Projekt initialisieren
railway init

# Deployen
railway up

# Domain generieren
railway domain
```

## 🔧 Wichtige Befehle

```bash
# Lokaler Server
streamlit run epa_dashboard2.py

# Requirements aktualisieren
pip freeze > requirements.txt

# Neue Dependencies installieren
pip install paket-name

# Git Status
git status

# Änderungen pushen
git add .
git commit -m "Deine Nachricht"
git push
```

## 📁 Projektstruktur

```
├── epa_dashboard2.py          ← Hauptdatei
├── requirements.txt           ← Python-Packages
├── Procfile                   ← Railway-Startbefehl
├── README.md                  ← Dokumentation
├── DEPLOYMENT.md              ← Deployment-Guide
├── .gitignore                 ← Git-Exclude
├── .streamlit/
│   └── config.toml           ← Streamlit-Config
└── Data/
    └── *.csv                 ← Daten hier
```

## ❓ Probleme?

### Dashboard startet nicht
```bash
# Requirements neu installieren
pip install -r requirements.txt --force-reinstall
```

### Railway-Fehler
```bash
# Logs anzeigen
railway logs
```

### Git-Probleme
```bash
# Cache leeren
git rm -r --cached .
git add .
git commit -m "Reset cache"
```

## 📞 Hilfe

- Vollständige Anleitung: `README.md`
- Deployment-Details: `DEPLOYMENT.md`
- GitHub Issues für Bugs

---

**Viel Erfolg! 🎉**
@echo off
REM Setup-Script für EPA Air Quality Dashboard
REM Windows Version

echo ================================
echo EPA Air Quality Dashboard Setup
echo ================================
echo.

REM Prüfen ob Python installiert ist
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python ist nicht installiert!
    echo Bitte installieren Sie Python 3.11 oder höher von python.org
    pause
    exit /b 1
)

echo [1/5] Python gefunden!
python --version

REM Virtuelle Umgebung erstellen
echo.
echo [2/5] Erstelle virtuelle Umgebung...
if exist .venv (
    echo Virtuelle Umgebung existiert bereits.
) else (
    python -m venv .venv
    echo Virtuelle Umgebung erstellt!
)

REM Virtuelle Umgebung aktivieren
echo.
echo [3/5] Aktiviere virtuelle Umgebung...
call .venv\Scripts\activate.bat

REM Dependencies installieren
echo.
echo [4/5] Installiere Dependencies...
echo Dies kann einige Minuten dauern...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Installation fehlgeschlagen!
    pause
    exit /b 1
)

echo.
echo [5/5] Prüfe Datenverzeichnis...
if not exist Data (
    mkdir Data
    echo Data-Ordner erstellt.
    echo HINWEIS: Bitte legen Sie die CSV-Datei in den Data-Ordner!
) else (
    echo Data-Ordner existiert.
)

echo.
echo ================================
echo Setup erfolgreich abgeschlossen!
echo ================================
echo.
echo Um das Dashboard zu starten:
echo   1. Aktiviere die virtuelle Umgebung: .venv\Scripts\activate
echo   2. Starte Streamlit: streamlit run epa_dashboard2.py
echo.
echo Oder führen Sie start_dashboard.bat aus
echo.
pause

