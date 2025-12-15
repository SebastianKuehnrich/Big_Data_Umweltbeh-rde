# 🌬️ EPA Air Quality Dashboard v2.0

Ein professionelles, interaktives Dashboard zur Visualisierung und Analyse von EPA-Luftqualitätsdaten für das Jahr 2024.

## 📋 Projektbeschreibung

Dieses Dashboard bietet umfassende Analysen und Visualisierungen der EPA (Environmental Protection Agency) Luftqualitätsdaten mit Fokus auf PM2.5-Messungen (Particulate Matter 2.5 µm) und Air Quality Index (AQI).

### ✨ Hauptfunktionen

- **📊 Echtzeit-Datenverarbeitung** mit DuckDB
- **🗺️ Multi-State Vergleich** - Vergleichen Sie mehrere Bundesstaaten gleichzeitig
- **📈 Interaktive Visualisierungen** mit Plotly
- **🔍 Anomalie-Erkennung** - Automatische Identifikation von Ausreißern
- **📉 Datenqualitäts-Monitoring** - Transparente Anzeige der Datenbereinigung
- **💾 Datenexport** - Download von Analysen und Rohdaten
- **🎨 Modernes Dark-Theme UI** - Optimiert für lange Arbeitssitzungen

### 🎯 Features

#### Dashboard-Tabs:
1. **Overview** - Zentrale KPIs und Schnellübersicht
2. **Detailed Analysis** - Zeitreihenanalysen und Trends
3. **State Comparison** - Bundesstaaten-Vergleich mit Heatmaps
4. **Data Quality** - Datenqualitätsmetriken und Bereinigungsstatistiken
5. **Raw Data Explorer** - Rohdaten-Browser mit Filteroptionen

## 🚀 Installation

### Voraussetzungen

- Python 3.11 oder höher
- pip (Python Package Manager)

### Lokale Installation

1. **Repository klonen:**
```bash
git clone https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git
cd Big_Data_Umweltbeh-rde
```

2. **Virtuelle Umgebung erstellen (empfohlen):**
```bash
python -m venv .venv
```

3. **Virtuelle Umgebung aktivieren:**

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

4. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

5. **Daten vorbereiten:**

Stellen Sie sicher, dass die Datei `Data/daily_88101_2024_cleaned.csv` vorhanden ist.
Falls Sie die Rohdaten haben, können Sie diese mit dem Cleanup-Script bereinigen.

6. **Dashboard starten:**
```bash
streamlit run epa_dashboard2.py
```

Das Dashboard öffnet sich automatisch im Browser unter `http://localhost:8501`

## 🚂 Deployment auf Railway

### Option 1: Über GitHub (empfohlen)

1. **Repository auf GitHub pushen:**
```bash
git init
git add .
git commit -m "Initial commit: EPA Air Quality Dashboard"
git branch -M main
git remote add origin https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git
git push -u origin main
```

2. **Railway Setup:**
   - Gehen Sie zu [railway.app](https://railway.app)
   - Klicken Sie auf "Start a New Project"
   - Wählen Sie "Deploy from GitHub repo"
   - Wählen Sie Ihr Repository aus
   - Railway erkennt automatisch die Streamlit-App

3. **Umgebungsvariablen (falls benötigt):**
   - In den Railway-Projekt-Einstellungen unter "Variables" können Sie zusätzliche Variablen setzen

4. **Deployment:**
   - Railway deployed automatisch nach jedem Push
   - Sie erhalten eine öffentliche URL für Ihr Dashboard

### Option 2: Railway CLI

1. **Railway CLI installieren:**
```bash
npm i -g @railway/cli
```

2. **In Railway einloggen:**
```bash
railway login
```

3. **Projekt initialisieren und deployen:**
```bash
railway init
railway up
```

## 📦 Projektstruktur

```
Big_Data_Umweltbeh-rde/
├── epa_dashboard2.py              # Haupt-Dashboard-Anwendung
├── CLEANUP_PIPELINE.py            # Daten-Bereinigungspipeline
├── requirements.txt               # Python-Dependencies
├── Procfile                       # Railway/Heroku Konfiguration
├── runtime.txt                   # Python-Version
├── railway.json                  # Railway-Konfiguration
├── README.md                     # Diese Datei
├── DEPLOYMENT.md                 # Deployment-Anleitung
├── QUICKSTART.md                 # Schnellstart-Guide
├── LICENSE                       # MIT-Lizenz
├── .gitignore                   # Git-Ignore-Regeln
├── .streamlit/
│   └── config.toml              # Streamlit-Konfiguration
├── setup.bat / setup.sh         # Setup-Scripts
├── start_dashboard.bat / .sh    # Start-Scripts
└── Data/
    └── daily_88101_2024_cleaned.csv  # Bereinigte EPA-Daten
```

## 🔧 Konfiguration

### Streamlit-Konfiguration

Die Datei `.streamlit/config.toml` enthält Theme- und Server-Einstellungen:

```toml
[theme]
primaryColor="#4ade80"
backgroundColor="#0e1117"
secondaryBackgroundColor="#262730"
textColor="#ffffff"
```

### Datenquelle anpassen

Um einen anderen Datensatz zu verwenden, ändern Sie den Pfad in der Hauptdatei:

```python
DATA_PATH = "Data/ihre_datei.csv"
```

## 📊 Datenformat

Das Dashboard erwartet CSV-Dateien mit folgenden Spalten:
- `Date Local` - Datum der Messung
- `State Name` - Name des Bundesstaates
- `County Name` - Name des Countys
- `Arithmetic Mean` - PM2.5-Wert (µg/m³)
- `AQI` - Air Quality Index

## 🛠️ Entwicklung

### Lokale Entwicklung mit Hot-Reload

```bash
streamlit run epa_dashboard2.py
```

Streamlit lädt die App automatisch neu, wenn Sie Änderungen speichern.

### Code-Struktur

Das Dashboard ist modular aufgebaut:

- **DataLoader**: Datenbankoperationen und Caching
- **Visualizations**: Chart-Erstellung mit Plotly
- **AnomalyDetector**: Statistische Anomalie-Erkennung
- **QueryBuilder**: SQL-Query-Generierung
- **EPADashboard**: Haupt-Anwendungsklasse

## 📈 Performance-Optimierung

- **DuckDB** für schnelle Datenverarbeitung
- **Streamlit Caching** (`@st.cache_data`) für wiederholte Abfragen
- **Lazy Loading** von großen Datensätzen
- **Optimierte SQL-Queries** mit CTEs

## 🐛 Troubleshooting

### Problem: "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Problem: "Data file not found"
Stellen Sie sicher, dass die CSV-Datei im `Data/`-Ordner liegt und der Pfad korrekt ist.

### Problem: Hoher Speicherverbrauch
- Reduzieren Sie den Datumsbereich in den Filtern
- Verwenden Sie die Parquet-Version der Daten (schneller und kompakter)

## 🤝 Beitragen

Contributions sind willkommen! Bitte:
1. Forken Sie das Repository
2. Erstellen Sie einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Committen Sie Ihre Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Pushen Sie zum Branch (`git push origin feature/AmazingFeature`)
5. Öffnen Sie einen Pull Request

## 📝 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt. Die EPA-Daten sind öffentlich verfügbar.

## 👤 Autor

**Sebastian Kühnrich**

- GitHub: [@SebastianKuehnrich](https://github.com/SebastianKuehnrich)
- Repository: [Big_Data_Umweltbeh-rde](https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde)

## 🙏 Danksagungen

- EPA für die öffentlichen Luftqualitätsdaten
- Streamlit Team für das großartige Framework
- DuckDB für die schnelle Datenverarbeitung
- Plotly für interaktive Visualisierungen

## 📞 Support

Bei Fragen oder Problemen öffnen Sie bitte ein Issue auf GitHub.

---

**Built with ❤️ and Python**

