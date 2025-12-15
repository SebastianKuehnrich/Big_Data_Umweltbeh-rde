# 📊 Tag 14: EPA Air Quality Dashboard

## Dein Montag-Projekt: Visualisiere deine Cleanup-Arbeit!

**Zeit:** 4-5 Stunden
**Schwierigkeit:** ⭐⭐⭐ (80% Bekannt, 20% Neu)
**Was ist neu:** Streamlit Framework

---

## 🎯 MISSION

Du hast am Wochenende **738.000 EPA Luftqualitätsdaten** bereinigt. Heute machst du diese Arbeit **sichtbar**!

Dein Chef sagt:

> *"Die Zahlen sind super, aber ich brauche ein Dashboard für das Management-Meeting morgen. Die wollen keine SQL-Tabellen sehen - die wollen Grafiken, Filter und einen 'Wow-Effekt'. Kannst du das bis heute Nachmittag?"*

---

## 📦 WAS IST STREAMLIT?

Streamlit ist ein Python-Framework für **interaktive Dashboards** - ohne HTML/CSS/JavaScript!

```python
import streamlit as st

st.title("Mein erstes Dashboard")
st.write("So einfach ist das!")
```

**Das war's.** Keine Frontend-Kenntnisse nötig.

### Installation (falls noch nicht installiert):
```bash
pip install streamlit
```

### Dashboard starten:
```bash
streamlit run dashboard.py
```

---

## 🏗️ PROJEKT-STRUKTUR

Du baust heute **5 Dashboard-Komponenten**:

| # | Komponente | Was du übst (80%) | Was neu ist (20%) |
|---|------------|-------------------|-------------------|
| 1 | KPI Cards | Aggregations-Queries | `st.metric()` |
| 2 | State Filter | WHERE Clause | `st.selectbox()` |
| 3 | Zeitfilter | Date Filtering | `st.date_input()` |
| 4 | AQI Trend Chart | GROUP BY Date | `st.line_chart()` |
| 5 | Cleanup Summary | CASE WHEN Counts | `st.bar_chart()` |

---

# TEIL 1: SETUP & ERSTE SCHRITTE (30 Min)

## 1.1 Erstelle deine Dashboard-Datei

Erstelle eine neue Datei: `epa_dashboard.py`

```python
# =============================================================================
# EPA AIR QUALITY DASHBOARD
# Tag 14 - Big Data Bootcamp
# =============================================================================

import streamlit as st
import duckdb
import pandas as pd

# -----------------------------------------------------------------------------
# KONFIGURATION
# -----------------------------------------------------------------------------

# Dein Dateipfad (ANPASSEN!)
DATA_PATH = 'C:/Users/DEIN_NAME/Documents/daily_88101_2024.csv'

# DuckDB Connection
@st.cache_resource
def get_connection():
    return duckdb.connect()

con = get_connection()

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="EPA Air Quality Dashboard",
    page_icon="🌬️",
    layout="wide"
)

st.title("🌬️ EPA Air Quality Dashboard 2024")
st.markdown("**Datenquelle:** EPA AirData | **Bereinigt mit:** COALESCE, GREATEST, LEAST")

# -----------------------------------------------------------------------------
# DEIN CODE KOMMT HIER...
# -----------------------------------------------------------------------------
```

## 1.2 Teste ob es funktioniert

```bash
streamlit run epa_dashboard.py
```

Du solltest einen Browser sehen mit deinem Titel. **Funktioniert?** Weiter geht's!

---

# TEIL 2: KPI CARDS - Die Übersicht (45 Min)

## Was du baust:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  738,478    │  │    53       │  │   2,755     │  │    7.11     │
│ Total Rows  │  │   States    │  │  Negative   │  │  Avg PM2.5  │
│             │  │             │  │   Fixed     │  │  (cleaned)  │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

## 2.1 Die Query (DU SCHREIBST!)

```python
# -----------------------------------------------------------------------------
# KPI METRICS
# -----------------------------------------------------------------------------

st.header("📈 Key Performance Indicators")

# Cache die Query für Performance
@st.cache_data
def get_kpi_data():
    return con.execute(f"""
        SELECT
            -- TODO: Zähle alle Zeilen
            -- TODO: Zähle einzigartige States
            -- TODO: Zähle negative PM2.5 Werte (die du gefixt hast)
            -- TODO: Berechne Durchschnitt PM2.5 (MIT CLEANUP FORMEL!)
            -- TODO: Zähle NULL AQI Werte (die du berechnet hast)
            -- TODO: Zähle extreme AQI > 500 (die du gekappt hast)
        FROM '{DATA_PATH}'
    """).fetchdf()

kpi_df = get_kpi_data()
```

**💡 Erinnerung - Die Cleanup Formel für PM2.5:**
```sql
GREATEST(COALESCE("Arithmetic Mean", 0), 0)
```

## 2.2 Die Anzeige (ICH GEB DIR DEN CODE)

```python
# Zeige KPIs in 4 Spalten
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📊 Total Measurements",
        value=f"{kpi_df['total_rows'].iloc[0]:,.0f}"
    )

with col2:
    st.metric(
        label="🗺️ US States Covered",
        value=f"{kpi_df['unique_states'].iloc[0]:,.0f}"
    )

with col3:
    st.metric(
        label="🔧 Negative Values Fixed",
        value=f"{kpi_df['negative_fixed'].iloc[0]:,.0f}",
        delta="Cleaned!",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="💨 Avg PM2.5 (Cleaned)",
        value=f"{kpi_df['avg_pm25'].iloc[0]:.2f} µg/m³"
    )
```

---

# TEIL 3: FILTER SIDEBAR (45 Min)

## Was du baust:

Eine Sidebar mit:
- State Dropdown (alle 53 Staaten)
- Datum von/bis Filter

## 3.1 State Liste holen (DU SCHREIBST!)

```python
# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------

st.sidebar.header("🔍 Filter")

# Hole alle States für Dropdown
@st.cache_data
def get_states():
    return con.execute(f"""
        -- TODO: Hole alle einzigartigen State Names
        -- TODO: Sortiere alphabetisch
        FROM '{DATA_PATH}'
    """).fetchdf()

states_df = get_states()
state_list = ['All States'] + states_df['State Name'].tolist()
```

## 3.2 Filter Widgets (ICH GEB DIR DEN CODE)

```python
# State Selector
selected_state = st.sidebar.selectbox(
    "🗺️ Select State",
    options=state_list,
    index=0
)

# Date Range
st.sidebar.subheader("📅 Date Range")

min_date = pd.to_datetime('2024-01-01')
max_date = pd.to_datetime('2024-12-31')

date_from = st.sidebar.date_input(
    "From",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

date_to = st.sidebar.date_input(
    "To",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)
```

## 3.3 Filter Query bauen (DU SCHREIBST!)

```python
# Baue die WHERE Clause dynamisch
def build_where_clause(state, date_from, date_to):
    conditions = []

    # TODO: Füge State-Bedingung hinzu (wenn nicht "All States")
    # Tipp: f'"State Name" = \'{state}\''

    # TODO: Füge Datum-Bedingung hinzu
    # Tipp: "Date Local" >= '{date_from}'

    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""

where_clause = build_where_clause(selected_state, date_from, date_to)
```

---

# TEIL 4: TREND CHART (60 Min)

## Was du baust:

Ein Liniendiagramm das zeigt:
- X-Achse: Monate 2024
- Y-Achse: Durchschnittliches PM2.5 (bereinigt!)

## 4.1 Monatliche Daten Query (DU SCHREIBST!)

```python
# -----------------------------------------------------------------------------
# TREND CHART
# -----------------------------------------------------------------------------

st.header("📈 Air Quality Trend 2024")

@st.cache_data
def get_monthly_trend(where_clause):
    return con.execute(f"""
        SELECT
            -- TODO: Extrahiere Jahr-Monat aus Date Local
            -- Tipp: STRFTIME("Date Local"::DATE, '%Y-%m') as month

            -- TODO: Berechne durchschnittliches PM2.5 (MIT CLEANUP!)

            -- TODO: Zähle "Unhealthy Days" (AQI > 100)

        FROM '{DATA_PATH}'
        {where_clause}
        -- TODO: Gruppiere nach Monat
        -- TODO: Sortiere nach Monat
    """).fetchdf()

trend_df = get_monthly_trend(where_clause)
```

## 4.2 Chart anzeigen (ICH GEB DIR DEN CODE)

```python
if not trend_df.empty:
    # Setze month als Index für den Chart
    chart_data = trend_df.set_index('month')

    # Zeige Line Chart
    st.line_chart(
        chart_data['avg_pm25'],
        use_container_width=True
    )

    # Zeige auch die Tabelle darunter
    with st.expander("📋 Show Data Table"):
        st.dataframe(trend_df, use_container_width=True)
else:
    st.warning("No data for selected filters")
```

---

# TEIL 5: CLEANUP SUMMARY CHART (45 Min)

## Was du baust:

Ein Balkendiagramm das zeigt wie viele Zeilen du bereinigt hast:
- Unchanged
- AQI Calculated
- Negative Corrected
- AQI Capped

## 5.1 Cleanup Categories Query (DU SCHREIBST!)

```python
# -----------------------------------------------------------------------------
# CLEANUP SUMMARY
# -----------------------------------------------------------------------------

st.header("🧹 Data Cleanup Summary")

@st.cache_data
def get_cleanup_summary(where_clause):
    return con.execute(f"""
        SELECT
            cleanup_action,
            COUNT(*) as count
        FROM (
            SELECT
                CASE
                    -- TODO: Kategorisiere jede Zeile
                    -- 'PM25 NULL replaced' wenn Arithmetic Mean IS NULL
                    -- 'Negative corrected' wenn Arithmetic Mean < 0
                    -- 'AQI capped at 500' wenn AQI > 500
                    -- 'AQI calculated' wenn AQI IS NULL
                    -- 'Unchanged' sonst
                END as cleanup_action
            FROM '{DATA_PATH}'
            {where_clause}
        )
        GROUP BY cleanup_action
        ORDER BY count DESC
    """).fetchdf()

cleanup_df = get_cleanup_summary(where_clause)
```

## 5.2 Bar Chart anzeigen (ICH GEB DIR DEN CODE)

```python
if not cleanup_df.empty:
    # Zwei Spalten: Chart links, Zahlen rechts
    col1, col2 = st.columns([2, 1])

    with col1:
        st.bar_chart(
            cleanup_df.set_index('cleanup_action'),
            use_container_width=True
        )

    with col2:
        st.subheader("📊 Numbers")
        for _, row in cleanup_df.iterrows():
            percentage = row['count'] / cleanup_df['count'].sum() * 100
            st.write(f"**{row['cleanup_action']}:** {row['count']:,} ({percentage:.2f}%)")
```

---

# TEIL 6: BONUS - TOP POLLUTED STATES TABLE (30 Min)

## Was du baust:

Eine sortierbare Tabelle der Staaten mit der schlechtesten Luftqualität.

## 6.1 Query (DU SCHREIBST!)

```python
# -----------------------------------------------------------------------------
# TOP POLLUTED STATES
# -----------------------------------------------------------------------------

st.header("🏭 States with Worst Air Quality")

@st.cache_data
def get_worst_states():
    return con.execute(f"""
        SELECT
            -- TODO: State Name
            -- TODO: Anzahl Messungen
            -- TODO: Durchschnitt PM2.5 (CLEANED!)
            -- TODO: Maximum PM2.5
            -- TODO: Anzahl Unhealthy Days (AQI > 100)
        FROM '{DATA_PATH}'
        -- TODO: Gruppiere nach State
        -- TODO: Sortiere nach avg_pm25 DESC
        -- TODO: LIMIT 10
    """).fetchdf()

worst_states_df = get_worst_states()
```

## 6.2 Tabelle anzeigen (ICH GEB DIR DEN CODE)

```python
st.dataframe(
    worst_states_df,
    use_container_width=True,
    column_config={
        "State Name": st.column_config.TextColumn("🗺️ State"),
        "measurements": st.column_config.NumberColumn("📊 Measurements", format="%d"),
        "avg_pm25": st.column_config.NumberColumn("💨 Avg PM2.5", format="%.2f"),
        "max_pm25": st.column_config.NumberColumn("⚠️ Max PM2.5", format="%.1f"),
        "unhealthy_days": st.column_config.NumberColumn("🔴 Unhealthy Days", format="%d")
    }
)
```

---

# TEIL 7: FINAL TOUCHES (30 Min)

## 7.1 Footer hinzufügen

```python
# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 EPA Air Quality Dashboard | Built with Streamlit & DuckDB</p>
        <p>Data Source: <a href='https://aqs.epa.gov/aqsweb/airdata/download_files.html'>EPA AirData</a></p>
        <p>Created by: [DEIN NAME] | Big Data Bootcamp Tag 14</p>
    </div>
""", unsafe_allow_html=True)
```

## 7.2 Sidebar Info

```python
# Am Ende der Sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
    **About this Dashboard**

    This dashboard visualizes 738,478 PM2.5 air quality
    measurements from the EPA for 2024.

    **Data Cleanup Applied:**
    - NULL values → Calculated
    - Negative values → Set to 0
    - Extreme AQI (>500) → Capped
""")
```

---

# 🧪 TESTING CHECKLIST

Bevor du fertig bist, teste:

- [ ] Dashboard startet ohne Fehler
- [ ] KPI Cards zeigen korrekte Zahlen
- [ ] State Filter funktioniert
- [ ] Date Filter funktioniert
- [ ] Trend Chart zeigt 12 Monate
- [ ] Cleanup Summary zeigt alle Kategorien
- [ ] Top States Tabelle ist sortiert

---

# 🚀 DEPLOYMENT (Optional aber Cool!)

## Streamlit Cloud (Kostenlos!)

1. Push deinen Code zu GitHub
2. Gehe zu [share.streamlit.io](https://share.streamlit.io)
3. Verbinde dein GitHub Repo
4. Deploy!

**Achtung:** Die CSV ist 284 MB - für Cloud musst du vielleicht eine kleinere Version verwenden.

---

# 📋 ABGABE

**Was du am Ende haben solltest:**

1. ✅ Funktionierendes Dashboard (`epa_dashboard.py`)
2. ✅ Alle 5 Komponenten implementiert
3. ✅ Filter funktionieren
4. ✅ Cleanup-Formeln korrekt angewendet
5. ✅ Screenshot für Präsentation morgen

---

# 💡 TROUBLESHOOTING

## "DuckDB kann Datei nicht finden"
```python
# Prüfe den Pfad
import os
print(os.path.exists(DATA_PATH))  # Sollte True sein
```

## "Cache Probleme"
```bash
# Lösche Streamlit Cache
streamlit cache clear
```

## "Port schon belegt"
```bash
# Nutze anderen Port
streamlit run dashboard.py --server.port 8502
```

## "Chart zeigt nichts"
- Prüfe ob deine Query Daten zurückgibt
- Füge `st.write(df)` ein zum Debuggen

---

# 🎯 ZUSAMMENFASSUNG

| Was du geübt hast (80%) | Was du gelernt hast (20%) |
|------------------------|---------------------------|
| DuckDB SQL Queries | `st.metric()` |
| COALESCE, GREATEST, LEAST | `st.selectbox()` |
| GROUP BY, Aggregations | `st.date_input()` |
| CASE WHEN | `st.line_chart()` |
| WHERE Clause Building | `st.bar_chart()` |
| Data Cleanup Pipeline | `st.columns()` |

---

**Viel Erfolg! 🚀**

*Bei Fragen → Erstmal Google/ChatGPT, dann Hand heben!*
