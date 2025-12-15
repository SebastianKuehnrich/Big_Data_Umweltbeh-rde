# 🚀 Deployment-Anleitung für Railway

## Schnellstart

### 1. GitHub-Repository vorbereiten

```bash
# Git initialisieren (falls noch nicht geschehen)
git init

# Alle Dateien hinzufügen
git add .

# Ersten Commit erstellen
git commit -m "Initial commit: EPA Air Quality Dashboard v2.0"

# Branch auf 'main' setzen
git branch -M main

# GitHub-Repository als Remote hinzufügen
git remote add origin https://github.com/SebastianKuehnrich/Big_Data_Umweltbeh-rde.git

# Zu GitHub pushen
git push -u origin main
```

### 2. Railway-Deployment

#### Option A: Über GitHub (Empfohlen)

1. **Railway-Account erstellen:**
   - Gehen Sie zu [railway.app](https://railway.app)
   - Registrieren Sie sich mit GitHub

2. **Neues Projekt erstellen:**
   - Klicken Sie auf "New Project"
   - Wählen Sie "Deploy from GitHub repo"
   - Wählen Sie Ihr `Big_Data_Umweltbeh-rde` Repository

3. **Automatisches Deployment:**
   - Railway erkennt automatisch die `Procfile`
   - Das Deployment startet automatisch
   - Nach ca. 2-3 Minuten ist die App verfügbar

4. **URL erhalten:**
   - Klicken Sie auf "Settings" → "Generate Domain"
   - Sie erhalten eine URL wie: `https://ihr-projekt.up.railway.app`

#### Option B: Mit Railway CLI

1. **Railway CLI installieren:**
```bash
npm install -g @railway/cli
```

2. **In Railway einloggen:**
```bash
railway login
```

3. **Projekt erstellen und deployen:**
```bash
# Neues Projekt initialisieren
railway init

# App hochladen und deployen
railway up

# Domain generieren
railway domain
```

### 3. Wichtige Dateien für Railway

Alle notwendigen Dateien sind bereits vorhanden:

- ✅ `requirements.txt` - Python-Dependencies
- ✅ `Procfile` - Startbefehl für Railway
- ✅ `runtime.txt` - Python-Version (3.11.7)
- ✅ `railway.json` - Railway-Konfiguration
- ✅ `.streamlit/config.toml` - Streamlit-Einstellungen

## ⚙️ Konfiguration

### Umgebungsvariablen (Optional)

Falls Sie Umgebungsvariablen benötigen:

1. Gehen Sie in Ihr Railway-Projekt
2. Klicken Sie auf "Variables"
3. Fügen Sie Variablen hinzu:

```bash
PORT=8501
STREAMLIT_SERVER_PORT=$PORT
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Port-Konfiguration

Railway weist automatisch einen Port zu. Der `Procfile` nutzt `$PORT`:

```
web: streamlit run epa_dashboard2.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

## 📊 Daten auf Railway

### Option 1: Daten im Repository (Aktuelle Lösung)

Die bereinigte CSV-Datei ist bereits im Repository:
```
Data/daily_88101_2024_cleaned.csv
```

**Vorteile:**
- ✅ Einfaches Deployment
- ✅ Keine zusätzliche Konfiguration nötig
- ✅ Daten sind immer verfügbar

**Nachteile:**
- ⚠️ Erhöht Repository-Größe
- ⚠️ Updates erfordern neuen Commit

### Option 2: Externe Datenquelle (Für große Dateien)

Für größere Datensätze können Sie:

1. **Cloud Storage verwenden:**
   - AWS S3
   - Google Cloud Storage
   - Railway Volumes

2. **Code anpassen:**
```python
# In epa_dashboard2.py
import os

# Für Railway Volume
DATA_PATH = os.getenv('DATA_PATH', 'Data/daily_88101_2024_cleaned.csv')
```

3. **Umgebungsvariable setzen:**
```bash
DATA_PATH=/app/data/your_file.csv
```

## 🔍 Troubleshooting

### Problem: "Application Error"

**Lösung 1 - Logs prüfen:**
```bash
railway logs
```

**Lösung 2 - Build neu triggern:**
```bash
git commit --allow-empty -m "Trigger rebuild"
git push
```

### Problem: "Module not found"

**Lösung:** `requirements.txt` prüfen:
```bash
# Alle Packages auflisten
pip freeze > requirements.txt

# Auf Railway neu deployen
git add requirements.txt
git commit -m "Update requirements"
git push
```

### Problem: Hoher Speicherverbrauch

**Lösung:** Railway-Plan upgraden oder Daten optimieren:

1. **Parquet statt CSV verwenden** (kleiner und schneller):
```python
# Konvertierung
import pandas as pd
df = pd.read_csv('data.csv')
df.to_parquet('data.parquet')
```

2. **DuckDB weiter optimieren:**
```python
# Indices erstellen
conn.execute("CREATE INDEX idx_date ON data('Date Local')")
```

### Problem: Port-Bindung

**Lösung:** Sicherstellen, dass Streamlit den richtigen Port verwendet:

```bash
# Im Procfile (bereits korrekt)
--server.port=$PORT
```

### Problem: Deployment dauert zu lange

**Mögliche Ursachen:**
1. Große Datendateien im Repository
2. Viele Dependencies

**Lösung:**
```bash
# .gitignore erweitern
echo "Data/*.csv" >> .gitignore

# Cache leeren und neu deployen
railway run --detach
```

## 📈 Performance-Tipps für Railway

### 1. Caching optimieren

```python
# In Streamlit mehr cachen
@st.cache_data(ttl=7200)  # 2 Stunden
def load_large_dataset():
    return pd.read_csv('data.csv')
```

### 2. DuckDB-Optimierungen

```python
# Verbindung mit mehr Memory
conn = duckdb.connect(':memory:', config={'memory_limit': '2GB'})
```

### 3. Streamlit-Config

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
enableXsrfProtection = false
enableCORS = false

[runner]
fastReruns = true
```

## 🔄 Continuous Deployment

Railway deployt automatisch bei jedem Push zu `main`:

```bash
# Änderungen machen
git add .
git commit -m "Update dashboard features"
git push

# Railway deployed automatisch
```

### Branch-Deployments

Für Test-Deployments:

```bash
# Feature-Branch erstellen
git checkout -b feature/new-viz

# Zu GitHub pushen
git push -u origin feature/new-viz

# In Railway: Settings → Connect to different branch
```

## 📊 Monitoring

### Railway Dashboard

- **Deployment-Status:** Live/Building/Failed
- **Logs:** Echtzeit-Logs der Anwendung
- **Metrics:** CPU, Memory, Network
- **Builds:** Build-History

### Logs anzeigen

```bash
# Via CLI
railway logs

# Im Browser
# Railway Dashboard → Ihr Projekt → Logs
```

## 💰 Kosten

Railway bietet:
- **Starter Plan:** $5/Monat + usage
- **Developer Plan:** $20/Monat
- **Team Plan:** $25/Monat/Mitglied

**Für dieses Dashboard empfohlen:**
- Starter Plan ist ausreichend für kleine bis mittlere Nutzung
- Ca. 100-500 MB RAM-Nutzung
- Minimale CPU-Last

## 🎯 Produktionsreife Checkliste

- [x] `requirements.txt` vollständig
- [x] `Procfile` konfiguriert
- [x] `.gitignore` gesetzt
- [x] `.streamlit/config.toml` optimiert
- [x] `README.md` dokumentiert
- [x] Datenbereinigung implementiert
- [x] Error Handling vorhanden
- [x] Caching aktiviert
- [ ] Custom Domain eingerichtet (optional)
- [ ] Analytics eingebunden (optional)
- [ ] Backup-Strategie definiert (optional)

## 🔗 Nützliche Links

- [Railway Dokumentation](https://docs.railway.app/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [DuckDB Dokumentation](https://duckdb.org/docs/)

## 📞 Support

Bei Railway-spezifischen Problemen:
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- Railway Docs: [docs.railway.app](https://docs.railway.app/)
- GitHub Issues: Für App-spezifische Probleme

---

**Viel Erfolg beim Deployment! 🚀**
# EPA Air Quality Dashboard - Dependencies
# Core Framework
streamlit==1.29.0

# Data Processing
duckdb==0.9.2
pandas==2.1.4
numpy==1.26.2

# Visualization
plotly==5.18.0

# Statistical Analysis
scipy==1.11.4

# Date/Time Handling
python-dateutil==2.8.2

# Utilities
typing-extensions==4.9.0

