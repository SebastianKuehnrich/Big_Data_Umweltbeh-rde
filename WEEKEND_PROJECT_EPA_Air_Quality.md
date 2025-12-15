# 🌬️ Weekend Project: EPA Air Quality Data Cleanup Pipeline

## Dein Real-World Data Engineering Challenge

**Geschätzte Zeit:** 4 Stunden (mit AI-Unterstützung für Debugging)
**Schwierigkeit:** ⭐⭐⭐ (Herausfordernd aber machbar!)
**Abgabe:** Montag vor dem Unterricht

---

## 🎯 DEINE MISSION

Du arbeitest als Junior Data Analyst bei einer Umweltbehörde. Dein Chef kommt am Freitagabend zu dir:

> *"Wir haben gerade **738.000 Luftqualitätsmessungen** von der EPA bekommen. Die Daten sind ein Chaos - NULL-Werte, negative Messwerte (unmöglich!), und ein Sensor in Massachusetts hat einen AQI von 1513 gemeldet... das Maximum ist 500! Ich brauche bis Montag einen sauberen Datensatz mit Analyse. Kannst du das?"*

**Dein Ziel:** Wende alles an, was du in Tag 13 gelernt hast - aber diesmal auf echten US-Regierungsdaten!

---

## 📊 DER DATENSATZ

**Quelle:** [EPA AirData](https://aqs.epa.gov/aqsweb/airdata/download_files.html) (data.gov)
**Datei:** `daily_88101_2024.csv` (PM2.5 Feinstaubmessungen)
**Größe:** ~284 MB, 738.478 Zeilen
**Zeitraum:** 1. Januar 2024 - 31. Dezember 2024
**Abdeckung:** Alle 50 US-Bundesstaaten + Puerto Rico

### Download-Anleitung

```python
# Option 1: Mit curl (im Terminal)
curl -L "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2024.zip" -o epa_pm25_2024.zip

# Option 2: Mit Python
import urllib.request
urllib.request.urlretrieve(
    "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2024.zip",
    "epa_pm25_2024.zip"
)

# Dann entpacken (Windows PowerShell):
# Expand-Archive -Path "epa_pm25_2024.zip" -DestinationPath "."
```

---

## 🔬 WAS IST PM2.5 UND AQI?

Bevor du loslegst - verstehe deine Daten!

### PM2.5 (Feinstaub)
- Partikel kleiner als 2.5 Mikrometer
- Kann tief in die Lunge eindringen
- Gemessen in **µg/m³** (Mikrogramm pro Kubikmeter)
- **Normale Werte:** 0-35 µg/m³
- **Gefährlich:** > 55 µg/m³

### AQI (Air Quality Index)
| AQI | Kategorie | Bedeutung |
|-----|-----------|-----------|
| 0-50 | Good | Luft ist sauber |
| 51-100 | Moderate | Akzeptabel |
| 101-150 | Unhealthy (Sensitive) | Risikogruppen betroffen |
| 151-200 | Unhealthy | Alle können Effekte spüren |
| 201-300 | Very Unhealthy | Gesundheitswarnung |
| 301-500 | Hazardous | Notfall! |

> 💡 **Wichtig:** AQI kann NIEMALS über 500 sein. Wenn du höhere Werte siehst = Sensorfehler!

---

## 📁 SETUP

```python
import duckdb
import pandas as pd
from datetime import datetime

# Dein Dateipfad (anpassen!)
DATA_PATH = 'Data/daily_88101_2024.csv'

# DuckDB Connection
con = duckdb.connect()

print("Let's go! 🚀")
```

---

# TEIL 1: DATA EXPLORATION (45 Min) 🔍

## Aufgabe 1.1: Dataset Overview

**Deine erste Query:** Finde heraus, was du vor dir hast.

```python
# STARTER CODE - Vervollständige die Query!
result = con.execute(f"""
    SELECT
        COUNT(*) as total_rows,
        -- Wie viele einzigartige Bundesstaaten?
        -- Wie viele einzigartige Counties?
        -- Was ist das früheste Datum?
        -- Was ist das späteste Datum?
    FROM '{DATA_PATH}'
""").fetchdf()
print(result.to_string(index=False))
```

**Erwartetes Ergebnis:**
```
total_rows: 738.478
unique_states: 53
date_range: 2024-01-01 bis 2024-12-31
```

---

## Aufgabe 1.2: Data Quality Audit

**Deine Aufgabe:** Zähle ALLE Datenqualitätsprobleme in EINER Query!

Zu finden:
| Problem | Spalte | Bedingung |
|---------|--------|-----------|
| NULL AQI | `AQI` | IS NULL |
| NULL Pollutant Standard | `Pollutant Standard` | IS NULL |
| NULL Site Name | `Local Site Name` | IS NULL |
| Negative PM2.5 | `Arithmetic Mean` | < 0 |
| Extreme AQI | `AQI` | > 500 |
| Low Observation % | `Observation Percent` | < 75 |

**💡 Hinweis:** Benutze `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` für jede Zählung!

```python
# DEIN CODE HIER
quality_audit = con.execute(f"""
    SELECT
        COUNT(*) as total_rows,
        -- Zähle jedes Problem...
    FROM '{DATA_PATH}'
""").fetchdf()
```

**🎯 Checkpoint:** Du solltest ungefähr finden:
- ~340.000 NULL AQI (46%)
- ~2.700 negative PM2.5 Werte
- Genau 2 extreme AQI Werte (>500)

---

## Aufgabe 1.3: Die Schuldigen finden

**Frage:** Welche Bundesstaaten haben die schlechteste Datenqualität?

```python
# Gruppiere nach "State Name" und finde:
# - Gesamtanzahl Messungen
# - Anzahl NULL AQI
# - Prozentsatz NULL
# - Anzahl negative PM2.5
# Sortiere nach Prozentsatz NULL (schlechteste zuerst)
```

**🔎 Was du entdecken wirst:**
- Einige Staaten haben ~50% NULL AQI!
- Missouri hat besonders viele negative Werte
- "Country Of Mexico" ist auch im Datensatz (Grenzstationen)

---

## Aufgabe 1.4: Outlier Investigation 🕵️

**KRITISCH:** Finde die extremen Ausreißer!

```python
# Finde alle Zeilen wo AQI > 500
# Zeige: State Name, County Name, Date, PM2.5 Wert, AQI, Site Name
```

**🚨 Was du finden wirst:**

| State | County | Date | PM2.5 | AQI | Was ist passiert? |
|-------|--------|------|-------|-----|-------------------|
| Massachusetts | Essex | 2024-11-10 | 833.8 | 1513 | ??? |
| California | San Bernardino | 2024-09-11 | 372.1 | 593 | ??? |

**Deine Aufgabe:** Recherchiere! Was ist an diesen Tagen passiert?
- Tipp für California: Google "Line Fire San Bernardino September 2024"
- Tipp für Massachusetts: Ist ein AQI von 1513 physikalisch möglich?

---

## Aufgabe 1.5: Negative Werte untersuchen

```python
# Finde die 15 negativsten PM2.5 Werte
# Zeige: State, County, Date, PM2.5 Wert, Sample Duration
# Sortiere von negativstem aufwärts
```

**❓ Denk nach:**
- Kann PM2.5 (Feinstaub in der Luft) negativ sein?
- Was bedeutet ein Wert von -7.06?
- Welche Cleanup-Funktion brauchst du dafür?

---

# TEIL 2: DATA CLEANUP PIPELINE (90 Min) 🧹

Jetzt wendest du an, was du in Tag 13 gelernt hast!

## Aufgabe 2.1: COALESCE - NULL AQI berechnen

**Das Problem:** 46% der Zeilen haben keinen AQI Wert!

**Die Lösung:** Berechne AQI aus PM2.5 mit der EPA-Formel (vereinfacht):

| PM2.5 Bereich | AQI Bereich | Formel |
|---------------|-------------|--------|
| 0 - 12.0 | 0 - 50 | PM2.5 * 50 / 12 |
| 12.1 - 35.4 | 51 - 100 | 50 + (PM2.5 - 12) * 49 / 23.4 |
| 35.5 - 55.4 | 101 - 150 | 100 + (PM2.5 - 35.4) * 49 / 20 |
| > 55.4 | > 150 | Weiter berechnen... |

```python
# STARTER CODE
coalesce_demo = con.execute(f"""
    SELECT
        "State Name",
        "Date Local",
        ROUND("Arithmetic Mean", 2) as pm25,
        "AQI" as aqi_original,
        COALESCE(
            "AQI",
            CASE
                WHEN "Arithmetic Mean" <= 12.0 THEN
                    -- Deine Formel hier...
                WHEN "Arithmetic Mean" <= 35.4 THEN
                    -- Deine Formel hier...
                -- Weitere Bereiche...
            END
        ) as aqi_calculated,
        CASE WHEN "AQI" IS NULL THEN 'Berechnet' ELSE 'Original' END as status
    FROM '{DATA_PATH}'
    WHERE "AQI" IS NULL
    LIMIT 15
""").fetchdf()
```

**🎯 Validierung:** Prüfe ob deine berechneten AQI-Werte sinnvoll sind!
- PM2.5 = 8.0 → AQI sollte ~33 sein
- PM2.5 = 25.0 → AQI sollte ~78 sein

---

## Aufgabe 2.2: GREATEST - Negative Werte korrigieren

**Das Problem:** ~2.700 negative PM2.5 Werte (Sensorfehler)

**Die Lösung:** `GREATEST(wert, 0)` - setzt alles Negative auf 0

```python
# Zeige 15 Beispiele der Korrektur
# Spalten: State, County, Date, pm25_original, pm25_fixed, correction_amount

# 💡 Hinweis: ABS() gibt dir den Absolutwert für die Differenz
```

**Erwartete Ausgabe:**
```
State     | County | Date       | pm25_original | pm25_fixed | correction
----------|--------|------------|---------------|------------|----------
Missouri  | Cedar  | 2024-05-01 | -7.06         | 0          | 7.06
...
```

---

## Aufgabe 2.3: LEAST - Extreme Werte kappen

**Das Problem:** AQI von 1513 ist unmöglich (Maximum ist 500)

**Die Lösung:** `LEAST(wert, 500)` - kappt bei 500

```python
# Finde ALLE Zeilen mit AQI > 300 (nicht nur > 500!)
# Zeige: State, County, Date, aqi_original, aqi_capped, amount_capped
# Füge eine Kategorie hinzu:
#   - > 1000: "SENSOR ERROR"
#   - > 500: "Capped"
#   - sonst: "Normal"
```

**🤔 Kritisches Denken:**
- Der California-Wert (593) wird auch gekappt. Ist das richtig?
- War das echte schlechte Luft oder ein Sensorfehler?
- Wie würdest du das in der Praxis entscheiden?

---

## Aufgabe 2.4: DIE MAGISCHE FORMEL 🪄

Kombiniere alles in EINER Formel!

**Für PM2.5:**
```sql
GREATEST(COALESCE("Arithmetic Mean", 0), 0)
```

**Für AQI:**
```sql
LEAST(GREATEST(COALESCE("AQI", berechneter_wert), 0), 500)
```

**Deine Aufgabe:** Erstelle eine Zusammenfassung der Cleanup-Aktionen:

```python
# Zähle wie viele Zeilen in jede Kategorie fallen:
# - "Unchanged" (keine Änderung nötig)
# - "AQI calculated" (NULL ersetzt)
# - "Negative corrected" (< 0 auf 0 gesetzt)
# - "AQI capped" (> 500 auf 500 gekappt)

# 💡 Hinweis: Nutze ein Subquery mit CASE WHEN, dann GROUP BY
```

**Erwartetes Ergebnis:**
```
cleanup_action      | count   | percentage
--------------------|---------|----------
Unchanged           | 395.464 | 53.55%
AQI calculated      | 340.257 | 46.08%
Negative corrected  | 2.755   | 0.37%
AQI capped          | 2       | 0.0003%
```

---

## Aufgabe 2.5: Data Quality Flags hinzufügen

Nicht alle Daten sind gleich zuverlässig! Erstelle Qualitäts-Flags basierend auf `Observation Percent`:

| Observation % | Flag |
|---------------|------|
| < 50% | UNRELIABLE |
| 50-74% | LOW_QUALITY |
| 75-89% | ACCEPTABLE |
| >= 90% | HIGH_QUALITY |

```python
# Zeige 15 Beispiele mit niedrigem Observation Percent
# Spalten: State, Date, PM2.5, AQI, Observation Percent, data_quality_flag
```

---

# TEIL 3: ANALYSE & INSIGHTS (60 Min) 📈

Jetzt wird's spannend - was sagen uns die Daten?

## Aufgabe 3.1: Luftqualitäts-Ranking der Bundesstaaten

**Frage:** Welche Staaten haben die schlechteste Luftqualität?

```python
# Berechne für jeden Staat:
# - Anzahl Messungen
# - Durchschnittliches PM2.5 (MIT CLEANUP!)
# - Maximum PM2.5
# - Durchschnittliches AQI
# Sortiere nach avg_pm25 DESC
# LIMIT 10
```

**🎯 Was du finden solltest:**
1. "Country Of Mexico" (Grenzstationen) ganz oben
2. Texas, Georgia, Arkansas mit hohen Werten
3. Warum haben bestimmte Staaten höhere Werte?

---

## Aufgabe 3.2: Saisonale Muster

**Frage:** Wann ist die Luft am schlechtesten?

```python
# Gruppiere nach Jahreszeit:
# Winter: Dezember, Januar, Februar
# Spring: März, April, Mai
# Summer: Juni, Juli, August
# Fall: September, Oktober, November

# 💡 Hinweis: MONTH("Date Local"::DATE) gibt dir den Monat als Zahl
```

**Erwartete Erkenntnis:**
- Welche Jahreszeit hat den höchsten Durchschnitt?
- In welcher Jahreszeit gibt es die meisten "Unhealthy Days" (AQI > 100)?
- Warum könnte das so sein? (Denk an Waldbrände, Heizung, etc.)

---

## Aufgabe 3.3: California Wildfire Deep Dive 🔥

**September 2024:** Die "Line Fire" in San Bernardino County war einer der größten Waldbrände des Jahres.

```python
# Analysiere California im September 2024
# Filter: State = 'California', Date enthält '2024-09', AQI > 150
# Zeige: County, Date, PM2.5, AQI, AQI_Category
# Sortiere nach AQI DESC
# LIMIT 20

# 💡 Für den Date-Filter:
# CAST("Date Local" AS VARCHAR) LIKE '2024-09%'
```

**🔎 Recherche-Aufgabe:**
- Wie viele Tage hatte San Bernardino "Unhealthy" oder schlimmere Luft?
- Welche anderen Counties waren betroffen?
- Korreliert das mit der Entfernung zum Feuer?

---

## Aufgabe 3.4: AQI Kategorien-Verteilung

**Frage:** Wie ist die Luftqualität in den USA insgesamt verteilt?

```python
# Erstelle eine Verteilung der AQI-Kategorien:
# 1. Good (0-50)
# 2. Moderate (51-100)
# 3. Unhealthy Sensitive (101-150)
# 4. Unhealthy (151-200)
# 5. Very Unhealthy (201-300)
# 6. Hazardous (301-500)

# Nutze die CLEANED AQI Werte!
```

**Erwartetes Ergebnis:**
```
category                    | count   | percentage
----------------------------|---------|----------
1. Good (0-50)              | 298.724 | 75.30%
2. Moderate (51-100)        | 96.736  | 24.38%
3. Unhealthy Sensitive      | 995     | 0.25%
...
```

---

## Aufgabe 3.5: Monatliche Trends

```python
# Erstelle einen monatlichen Überblick:
# - Anzahl Messungen
# - Durchschnittliches PM2.5 (cleaned)
# - Maximum PM2.5
# - Anzahl "Unhealthy Days" (AQI > 100)

# 💡 Hinweis: STRFTIME("Date Local"::DATE, '%Y-%m') für Monat
```

**📊 Visualisierungsidee:**
Exportiere das Ergebnis und erstelle ein Liniendiagramm!

---

# TEIL 4: VALIDIERUNG & DOKUMENTATION (45 Min) ✅

## Aufgabe 4.1: Vorher vs. Nachher Vergleich

**DER WICHTIGSTE BEWEIS!** Zeige, dass dein Cleanup funktioniert hat.

```python
# Erstelle eine UNION ALL Query mit:
# VORHER: Analyse der Rohdaten
# NACHHER: Analyse mit Cleanup-Formeln

# Metriken:
# - total_rows
# - null_pm25 (sollte NACHHER: 0 sein)
# - negative_pm25 (sollte NACHHER: 0 sein)
# - extreme_aqi (sollte NACHHER: 0 sein)
# - avg_pm25
# - min_pm25 (VORHER: negativ, NACHHER: 0)
# - max_pm25
# - min_aqi
# - max_aqi (VORHER: 1513, NACHHER: 500)
```

**Erwartete Ausgabe:**
```
status  | total   | null_pm25 | negative | extreme | min_pm25 | max_aqi
--------|---------|-----------|----------|---------|----------|--------
VORHER  | 738.478 | 0         | 2.755    | 2       | -7.06    | 1513
NACHHER | 738.478 | 0         | 0        | 0       | 0.00     | 500
```

---

## Aufgabe 4.2: Cleaned Dataset Export (Sample)

```python
# Erstelle einen Export mit allen Cleanup-Transformationen:
# Spalten:
# - State Name
# - County Name
# - Date Local
# - Sample Duration
# - pm25_cleaned (mit GREATEST + COALESCE)
# - aqi_cleaned (mit LEAST + GREATEST + COALESCE + Berechnung)
# - data_quality_flag
# - Latitude
# - Longitude

# LIMIT 10 für Sample
```

---

## Aufgabe 4.3: Executive Summary

Schreibe eine kurze Zusammenfassung (5-10 Sätze) mit:

1. **Dataset:** Wie viele Zeilen, welcher Zeitraum, welche Abdeckung?
2. **Probleme gefunden:** Was waren die Hauptprobleme?
3. **Cleanup durchgeführt:** Welche Techniken hast du angewendet?
4. **Ergebnisse:** Was hat der Cleanup erreicht?
5. **Key Insights:** Was hast du über die US-Luftqualität gelernt?

---

# 🏆 BONUS-CHALLENGES (Optional)

Wenn du noch Zeit hast...

## Bonus 1: Worst Air Quality Day 2024
Finde den Tag mit der schlechtesten durchschnittlichen Luftqualität in den gesamten USA.

## Bonus 2: Geographic Hotspots
Welche 5 Messstationen hatten die meisten "Unhealthy Days"?

## Bonus 3: Weekend vs. Weekday
Ist die Luftqualität am Wochenende besser als unter der Woche? (Weniger Verkehr?)

---

# 🔧 TROUBLESHOOTING

## Library Versions (getestet!)

| Library | Version | Install |
|---------|---------|---------|
| Python | 3.13+ | - |
| duckdb | 1.4.2 | `pip install duckdb==1.4.2` |
| pandas | 2.3.1 | `pip install pandas==2.3.1` |
| numpy | 2.2.6 | `pip install numpy==2.2.6` |
| pyarrow | 19.0.1 | `pip install pyarrow==19.0.1` |

**Quick Install:**
```bash
pip install duckdb==1.4.2 pandas==2.3.1 numpy==2.2.6 pyarrow==19.0.1
```

## Häufige Fehler

### "LIKE on DATE column" Error
```sql
-- FALSCH:
WHERE "Date Local" LIKE '2024-09%'

-- RICHTIG:
WHERE CAST("Date Local" AS VARCHAR) LIKE '2024-09%'
```

### "Column not found" Error
```sql
-- Spaltennamen mit Leerzeichen brauchen Anführungszeichen!
"State Name"    -- RICHTIG
State Name      -- FALSCH
```

### Große Datei lädt langsam
```python
# Erste Tests mit LIMIT
con.execute(f"SELECT * FROM '{DATA_PATH}' LIMIT 1000")
```

---

# 📋 ABGABE-CHECKLISTE

- [ ] **Teil 1:** Alle Datenqualitätsprobleme identifiziert
- [ ] **Teil 2:** COALESCE, GREATEST, LEAST korrekt angewendet
- [ ] **Teil 3:** Mindestens 3 Analysen durchgeführt
- [ ] **Teil 4:** Vorher/Nachher Vergleich zeigt Erfolg
- [ ] **Bonus:** Mindestens 1 Bonus-Challenge (optional)
- [ ] **Code:** Läuft ohne Fehler durch
- [ ] **Summary:** Executive Summary geschrieben

---

# 💡 FINAL HINTS

1. **Test small first:** Nutze LIMIT 100 beim Entwickeln
2. **Step by step:** Eine Funktion nach der anderen testen
3. **Print often:** Zwischenergebnisse ausgeben
4. **AI is allowed:** Für Debugging und Syntax-Hilfe
5. **Think critically:** Warum sind die Daten so wie sie sind?

---

**Viel Erfolg! Du schaffst das! 🚀**

*Bei Fragen: Notiere sie und bring sie Montag mit!*
