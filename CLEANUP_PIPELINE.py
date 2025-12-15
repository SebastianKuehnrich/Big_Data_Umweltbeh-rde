"""
🌬️ EPA AIR QUALITY DATA CLEANUP PIPELINE - ENHANCED VERSION
=============================================================
Professional Data Engineering Pipeline with OOP Architecture

Features:
- Object-oriented design with configuration management
- Comprehensive data quality reporting
- Flexible cleanup strategies
- Before/After validation
- Multiple export formats
- Detailed logging and examples

Author: Data Engineering Student
Date: December 2024
Dataset: EPA AirData PM2.5 Measurements 2024
"""

import duckdb
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EPACleanupConfig:
    """Zentrale Konfiguration für EPA Data Cleanup Pipeline"""

    # Qualitäts-Schwellwerte
    aqi_max: int = 500  # EPA Maximum
    pm25_min: float = 0.0  # Physikalisches Minimum
    pm25_extreme_threshold: float = 150.0  # Warnung bei > 150 µg/m³
    observation_percent_min: float = 75.0  # Minimum für HIGH_QUALITY

    # Cleanup-Strategien
    null_aqi_strategy: str = "calculate"  # "calculate" oder "zero"
    fix_negative_pm25: bool = True  # Negative auf 0 setzen?
    cap_extreme_aqi: bool = True  # AQI auf 500 kappen?

    # Ausgabe-Einstellungen
    show_examples: bool = True  # Beispiele anzeigen?
    example_limit: int = 10  # Max. Beispiele pro Kategorie
    verbose: bool = True  # Ausführliche Ausgabe?

    # Export-Einstellungen
    export_format: str = "parquet"  # "parquet", "csv", oder "both"
    include_cleanup_flags: bool = True  # Cleanup-Tracking-Spalten?


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class EPADataQualityPipeline:
    """
    Professionelle Data Quality Pipeline für EPA Air Quality Daten

    Phasen:
    0. Setup & Validierung
    1. Schema-Analyse
    2. Qualitäts-Statistiken (VORHER)
    3. Problem-Beispiele
    4. Data Cleanup
    5. Validation (NACHHER)
    6. Export
    """

    def __init__(self, data_path: str, config: Optional[EPACleanupConfig] = None):
        """
        Initialisiert die Pipeline

        Args:
            data_path: Pfad zur CSV-Datei
            config: Optional custom configuration
        """
        self.config = config or EPACleanupConfig()
        self.data_path = self._validate_path(data_path)
        self.con = None
        self.stats_before = None
        self.stats_after = None

    def _validate_path(self, path: str) -> Path:
        """Validiert und normalisiert Dateipfad"""
        path_obj = Path(path)

        # Absoluter Pfad
        if path_obj.is_absolute() and path_obj.exists():
            return path_obj

        # Relativer Pfad
        if path_obj.exists():
            return path_obj.resolve()

        # Relativ zum Script-Verzeichnis
        script_dir = Path(__file__).parent
        relative_path = script_dir / path
        if relative_path.exists():
            return relative_path

        # Parent-Verzeichnis versuchen
        parent_path = script_dir.parent / path
        if parent_path.exists():
            return parent_path

        raise FileNotFoundError(
            f"❌ Datei nicht gefunden: {path}\n"
            f"   Getestet:\n"
            f"   - {path_obj}\n"
            f"   - {relative_path}\n"
            f"   - {parent_path}\n"
        )

    def _connect_db(self) -> None:
        """Erstellt DuckDB-Verbindung und validiert Datenzugriff"""
        try:
            self.con = duckdb.connect()

            # Test-Query
            row_count = self.con.execute(
                f"SELECT COUNT(*) FROM '{self.data_path}'"
            ).fetchone()[0]

            if self.config.verbose:
                print(f"✅ Datei erfolgreich geladen: {row_count:,} Zeilen")

        except Exception as e:
            raise RuntimeError(f"❌ DuckDB-Verbindung fehlgeschlagen: {e}")

    # ========================================================================
    # PHASE 1: SCHEMA-ANALYSE
    # ========================================================================

    def analyze_schema(self) -> pd.DataFrame:
        """Analysiert Datentypen und Struktur"""

        if not self.config.verbose:
            return None

        print("\n" + "=" * 80)
        print("📋 PHASE 1: SCHEMA & DATENTYPEN")
        print("=" * 80)

        # Schema abrufen
        schema = self.con.execute(
            f"DESCRIBE SELECT * FROM '{self.data_path}'"
        ).fetchdf()

        print("\n🔹 Spalten-Schema (wichtigste Felder):")
        important_cols = [
            'State Name', 'County Name', 'Date Local',
            'Arithmetic Mean', 'AQI', 'Observation Percent'
        ]

        for col in important_cols:
            col_info = schema[schema['column_name'] == col]
            if not col_info.empty:
                print(f"   • {col:30s} → {col_info.iloc[0]['column_type']}")

        # Typ-Validierung
        print("\n🔹 Datentyp-Check:")
        type_check = self.con.execute(f"""
            SELECT
                TYPEOF("Arithmetic Mean") as pm25_type,
                TYPEOF("AQI") as aqi_type,
                TYPEOF("Date Local") as date_type,
                TYPEOF("Observation Percent") as obs_pct_type
            FROM '{self.data_path}'
            LIMIT 1
        """).fetchdf()
        print(type_check.to_string(index=False))

        return schema

    # ========================================================================
    # PHASE 2: QUALITÄTS-STATISTIKEN (VORHER)
    # ========================================================================

    def calculate_quality_stats(self) -> Dict:
        """Berechnet umfassende Datenqualitäts-Metriken"""

        print("\n" + "=" * 80)
        print("📊 PHASE 2: DATENQUALITÄTS-AUDIT (VORHER)")
        print("=" * 80)

        stats = self.con.execute(f"""
            SELECT
                COUNT(*) as total_rows,

                -- NULL-Werte
                SUM(CASE WHEN "State Name" IS NULL THEN 1 ELSE 0 END) as null_state,
                SUM(CASE WHEN "Arithmetic Mean" IS NULL THEN 1 ELSE 0 END) as null_pm25,
                SUM(CASE WHEN "AQI" IS NULL THEN 1 ELSE 0 END) as null_aqi,
                SUM(CASE WHEN "Pollutant Standard" IS NULL THEN 1 ELSE 0 END) as null_standard,
                SUM(CASE WHEN "Local Site Name" IS NULL THEN 1 ELSE 0 END) as null_site,

                -- Wertebereich-Probleme: PM2.5
                SUM(CASE WHEN "Arithmetic Mean" < 0 THEN 1 ELSE 0 END) as negative_pm25,
                SUM(CASE WHEN "Arithmetic Mean" > {self.config.pm25_extreme_threshold} 
                    THEN 1 ELSE 0 END) as extreme_pm25,

                -- Wertebereich-Probleme: AQI
                SUM(CASE WHEN "AQI" < 0 THEN 1 ELSE 0 END) as negative_aqi,
                SUM(CASE WHEN "AQI" > {self.config.aqi_max} THEN 1 ELSE 0 END) as extreme_aqi,

                -- Datenqualität: Observation Percent
                SUM(CASE WHEN "Observation Percent" < 50 THEN 1 ELSE 0 END) as unreliable_obs,
                SUM(CASE WHEN "Observation Percent" >= 50 AND "Observation Percent" < 75 
                    THEN 1 ELSE 0 END) as low_quality_obs,
                SUM(CASE WHEN "Observation Percent" >= 75 AND "Observation Percent" < 90 
                    THEN 1 ELSE 0 END) as acceptable_obs,
                SUM(CASE WHEN "Observation Percent" >= 90 THEN 1 ELSE 0 END) as high_quality_obs,

                -- Statistiken: PM2.5
                ROUND(AVG("Arithmetic Mean"), 2) as avg_pm25,
                ROUND(MEDIAN("Arithmetic Mean"), 2) as median_pm25,
                ROUND(MIN("Arithmetic Mean"), 2) as min_pm25,
                ROUND(MAX("Arithmetic Mean"), 2) as max_pm25,

                -- Statistiken: AQI
                ROUND(AVG("AQI"), 2) as avg_aqi,
                ROUND(MEDIAN("AQI"), 2) as median_aqi,
                MIN("AQI") as min_aqi,
                MAX("AQI") as max_aqi,

                -- Zeitraum
                MIN("Date Local") as earliest_date,
                MAX("Date Local") as latest_date,
                COUNT(DISTINCT "State Name") as unique_states,
                COUNT(DISTINCT "County Name") as unique_counties

            FROM '{self.data_path}'
        """).fetchdf()

        stats_dict = stats.to_dict('records')[0]
        self.stats_before = stats_dict

        # Ausgabe formatiert
        self._print_quality_stats(stats_dict, "VORHER")

        return stats_dict

    def _print_quality_stats(self, stats: Dict, label: str) -> None:
        """Formatierte Ausgabe der Qualitäts-Statistiken"""

        total = stats['total_rows']

        print(f"\n📈 DATENSATZ-ÜBERSICHT:")
        print(f"   • Gesamt-Zeilen:    {total:,}")
        print(f"   • Zeitraum:         {stats.get('earliest_date', 'N/A')} bis {stats.get('latest_date', 'N/A')}")
        print(f"   • Bundesstaaten:    {stats.get('unique_states', 'N/A')}")
        print(f"   • Counties:         {stats.get('unique_counties', 'N/A')}")

        print(f"\n🔴 NULL-WERTE:")
        print(f"   • Fehlende States:         {stats['null_state']:,} ({stats['null_state'] / total * 100:.1f}%)")
        print(f"   • Fehlende PM2.5:          {stats['null_pm25']:,} ({stats['null_pm25'] / total * 100:.1f}%)")
        print(f"   • Fehlende AQI:            {stats['null_aqi']:,} ({stats['null_aqi'] / total * 100:.1f}%)")
        print(f"   • Fehlende Standards:      {stats['null_standard']:,} ({stats['null_standard'] / total * 100:.1f}%)")
        print(f"   • Fehlende Site Names:     {stats['null_site']:,} ({stats['null_site'] / total * 100:.1f}%)")

        print(f"\n🔴 WERTEBEREICH-PROBLEME:")
        print(f"   PM2.5:")
        print(f"   • Negative Werte:          {stats['negative_pm25']:,}")
        print(f"   • Extreme Werte (>{self.config.pm25_extreme_threshold}):  {stats['extreme_pm25']:,}")
        print(f"   AQI:")
        print(f"   • Negative Werte:          {stats['negative_aqi']:,}")
        print(f"   • Über Maximum (>{self.config.aqi_max}):      {stats['extreme_aqi']:,}")

        print(f"\n🟡 DATENQUALITÄT (Observation %):")
        print(
            f"   • UNRELIABLE (<50%):       {stats['unreliable_obs']:,} ({stats['unreliable_obs'] / total * 100:.1f}%)")
        print(
            f"   • LOW_QUALITY (50-74%):    {stats['low_quality_obs']:,} ({stats['low_quality_obs'] / total * 100:.1f}%)")
        print(
            f"   • ACCEPTABLE (75-89%):     {stats['acceptable_obs']:,} ({stats['acceptable_obs'] / total * 100:.1f}%)")
        print(
            f"   • HIGH_QUALITY (≥90%):     {stats['high_quality_obs']:,} ({stats['high_quality_obs'] / total * 100:.1f}%)")

        print(f"\n📊 VERTEILUNGS-KONTEXT:")
        print(f"   PM2.5 (µg/m³):")
        print(f"   • Durchschnitt: {stats['avg_pm25']} | Median: {stats['median_pm25']}")
        print(f"   • Min/Max:      {stats['min_pm25']} / {stats['max_pm25']}")
        print(f"   AQI:")
        print(f"   • Durchschnitt: {stats['avg_aqi']} | Median: {stats['median_aqi']}")
        print(f"   • Min/Max:      {stats['min_aqi']} / {stats['max_aqi']}")

    # ========================================================================
    # PHASE 3: PROBLEM-BEISPIELE
    # ========================================================================

    def show_problem_examples(self) -> None:
        """Zeigt konkrete Beispiele für jedes Datenproblem"""

        if not self.config.show_examples:
            return

        print("\n" + "=" * 80)
        print(f"🔍 PHASE 3: PROBLEM-BEISPIELE (max. {self.config.example_limit} pro Kategorie)")
        print("=" * 80)

        problems = [
            {
                "name": "NULL AQI (fehlende Werte)",
                "condition": '"AQI" IS NULL',
                "columns": '"State Name", "County Name", "Date Local", "Arithmetic Mean", "AQI"'
            },
            {
                "name": "Negative PM2.5 (Sensorfehler)",
                "condition": '"Arithmetic Mean" < 0',
                "columns": '"State Name", "County Name", "Date Local", "Arithmetic Mean", "Sample Duration"'
            },
            {
                "name": f"Extreme AQI (>{self.config.aqi_max})",
                "condition": f'"AQI" > {self.config.aqi_max}',
                "columns": '"State Name", "County Name", "Date Local", "Arithmetic Mean", "AQI"'
            },
            {
                "name": f"Extreme PM2.5 (>{self.config.pm25_extreme_threshold} µg/m³)",
                "condition": f'"Arithmetic Mean" > {self.config.pm25_extreme_threshold}',
                "columns": '"State Name", "County Name", "Date Local", "Arithmetic Mean", "AQI"'
            },
            {
                "name": "Niedrige Observation % (<50%)",
                "condition": '"Observation Percent" < 50',
                "columns": '"State Name", "Date Local", "Observation Percent", "Arithmetic Mean", "AQI"'
            }
        ]

        for problem in problems:
            count = self.con.execute(f"""
                SELECT COUNT(*) FROM '{self.data_path}' WHERE {problem['condition']}
            """).fetchone()[0]

            if count > 0:
                print(f"\n❌ {problem['name']} ({count:,} Zeilen)")
                print("-" * 80)

                examples = self.con.execute(f"""
                    SELECT {problem['columns']}
                    FROM '{self.data_path}'
                    WHERE {problem['condition']}
                    ORDER BY RANDOM()
                    LIMIT {self.config.example_limit}
                """).fetchdf()

                print(examples.to_string(index=False))
            else:
                print(f"\n✅ {problem['name']}: Keine Probleme gefunden")

    # ========================================================================
    # PHASE 4: DATA CLEANUP
    # ========================================================================

    def _calculate_aqi_from_pm25(self, pm25_col: str = '"Arithmetic Mean"') -> str:
        """
        Generiert EPA AQI-Berechnungsformel für PM2.5

        EPA Breakpoints für PM2.5:
        0-12.0    → AQI 0-50
        12.1-35.4  → AQI 51-100
        35.5-55.4  → AQI 101-150
        55.5-150.4 → AQI 151-200
        150.5-250.4 → AQI 201-300
        250.5-500.4 → AQI 301-500
        """
        return f"""
            CASE
                WHEN {pm25_col} <= 12.0 THEN
                    {pm25_col} * 50.0 / 12.0
                WHEN {pm25_col} <= 35.4 THEN
                    50 + ({pm25_col} - 12.0) * 49.0 / 23.4
                WHEN {pm25_col} <= 55.4 THEN
                    100 + ({pm25_col} - 35.4) * 49.0 / 20.0
                WHEN {pm25_col} <= 150.4 THEN
                    150 + ({pm25_col} - 55.4) * 49.0 / 95.0
                WHEN {pm25_col} <= 250.4 THEN
                    200 + ({pm25_col} - 150.4) * 99.0 / 100.0
                ELSE
                    300 + ({pm25_col} - 250.4) * 199.0 / 250.0
            END
        """

    def apply_cleanup(self) -> None:
        """Führt kompletten Data Cleanup durch"""

        print("\n" + "=" * 80)
        print("🔧 PHASE 4: DATA CLEANUP")
        print("=" * 80)

        print("\n🔹 Angewandte Cleanup-Regeln:")
        print(f"   • NULL AQI → {self.config.null_aqi_strategy}")
        print(f"   • Negative PM2.5 → {'auf 0 setzen' if self.config.fix_negative_pm25 else 'unverändert'}")
        print(
            f"   • Extreme AQI → {'auf ' + str(self.config.aqi_max) + ' kappen' if self.config.cap_extreme_aqi else 'unverändert'}")

        # PM2.5 Cleanup-Formel
        pm25_formula = '"Arithmetic Mean"'
        if self.config.fix_negative_pm25:
            pm25_formula = f'GREATEST({pm25_formula}, 0)'

        # AQI Cleanup-Formel
        if self.config.null_aqi_strategy == "calculate":
            aqi_calculated = self._calculate_aqi_from_pm25(pm25_formula)
            aqi_formula = f'COALESCE("AQI", {aqi_calculated})'
        else:
            aqi_formula = 'COALESCE("AQI", 0)'

        # Extreme AQI kappen
        if self.config.cap_extreme_aqi:
            aqi_formula = f'LEAST({aqi_formula}, {self.config.aqi_max})'

        # Stelle sicher, dass AQI nicht negativ ist
        aqi_formula = f'GREATEST({aqi_formula}, 0)'

        # Quality Flag erstellen
        quality_flag = f"""
            CASE
                WHEN "Observation Percent" < 50 THEN 'UNRELIABLE'
                WHEN "Observation Percent" < 75 THEN 'LOW_QUALITY'
                WHEN "Observation Percent" < 90 THEN 'ACCEPTABLE'
                ELSE 'HIGH_QUALITY'
            END
        """

        # Cleanup Action Tracking
        cleanup_action = f"""
            CASE
                WHEN "AQI" IS NULL THEN 'AQI_CALCULATED'
                WHEN "Arithmetic Mean" < 0 THEN 'NEGATIVE_FIXED'
                WHEN "AQI" > {self.config.aqi_max} THEN 'AQI_CAPPED'
                WHEN "Arithmetic Mean" > {self.config.pm25_extreme_threshold} THEN 'EXTREME_PM25'
                ELSE 'UNCHANGED'
            END
        """

        # View erstellen
        self.con.execute(f"""
            CREATE OR REPLACE TEMP VIEW cleaned_data AS
            SELECT 
                "State Name",
                "County Name",
                "Date Local",
                "Local Site Name",
                "Latitude",
                "Longitude",
                "Sample Duration",
                "Pollutant Standard",
                "Observation Percent",

                -- Original-Werte (für Vergleich)
                "Arithmetic Mean" as pm25_original,
                "AQI" as aqi_original,

                -- Bereinigte Werte
                ROUND({pm25_formula}, 2) as pm25_cleaned,
                ROUND({aqi_formula}, 0) as aqi_cleaned,

                -- Quality Flags
                {quality_flag} as data_quality_flag,
                {cleanup_action} as cleanup_action

            FROM '{self.data_path}'
        """)

        print("\n✅ Cleaned Data View erstellt")

        # Cleanup-Statistik
        cleanup_summary = self.con.execute("""
            SELECT 
                cleanup_action,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percent
            FROM cleaned_data
            GROUP BY cleanup_action
            ORDER BY count DESC
        """).fetchdf()

        print("\n🔹 Cleanup-Zusammenfassung:")
        print(cleanup_summary.to_string(index=False))

        # Quality Flag Distribution
        quality_distribution = self.con.execute("""
            SELECT 
                data_quality_flag,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percent
            FROM cleaned_data
            GROUP BY data_quality_flag
            ORDER BY 
                CASE data_quality_flag
                    WHEN 'HIGH_QUALITY' THEN 1
                    WHEN 'ACCEPTABLE' THEN 2
                    WHEN 'LOW_QUALITY' THEN 3
                    ELSE 4
                END
        """).fetchdf()

        print("\n🔹 Datenqualitäts-Verteilung:")
        print(quality_distribution.to_string(index=False))

    # ========================================================================
    # PHASE 5: VALIDATION
    # ========================================================================

    def validate_cleanup(self) -> Tuple[Dict, Dict]:
        """Validiert Cleanup-Erfolg durch Vorher-Nachher-Vergleich"""

        print("\n" + "=" * 80)
        print("✅ PHASE 5: VORHER-NACHHER-VALIDATION")
        print("=" * 80)

        # Statistiken NACHHER
        stats_after = self.con.execute(f"""
            SELECT
                COUNT(*) as total_rows,

                -- NULL-Werte (sollten weg sein)
                SUM(CASE WHEN pm25_cleaned IS NULL THEN 1 ELSE 0 END) as null_pm25,
                SUM(CASE WHEN aqi_cleaned IS NULL THEN 1 ELSE 0 END) as null_aqi,

                -- Wertebereich-Probleme (sollten weg sein)
                SUM(CASE WHEN pm25_cleaned < 0 THEN 1 ELSE 0 END) as negative_pm25,
                SUM(CASE WHEN aqi_cleaned < 0 THEN 1 ELSE 0 END) as negative_aqi,
                SUM(CASE WHEN aqi_cleaned > {self.config.aqi_max} THEN 1 ELSE 0 END) as extreme_aqi,

                -- Statistiken
                ROUND(AVG(pm25_cleaned), 2) as avg_pm25,
                ROUND(MEDIAN(pm25_cleaned), 2) as median_pm25,
                ROUND(MIN(pm25_cleaned), 2) as min_pm25,
                ROUND(MAX(pm25_cleaned), 2) as max_pm25,

                ROUND(AVG(aqi_cleaned), 2) as avg_aqi,
                ROUND(MEDIAN(aqi_cleaned), 2) as median_aqi,
                MIN(aqi_cleaned) as min_aqi,
                MAX(aqi_cleaned) as max_aqi

            FROM cleaned_data
        """).fetchdf().to_dict('records')[0]

        self.stats_after = stats_after

        # Vergleichstabelle erstellen
        print("\n📊 VORHER-NACHHER-VERGLEICH:")
        print("-" * 80)

        comparison = self.con.execute(f"""
            SELECT
                'VORHER' as status,
                COUNT(*) as total,
                SUM(CASE WHEN "Arithmetic Mean" IS NULL THEN 1 ELSE 0 END) as null_pm25,
                SUM(CASE WHEN "AQI" IS NULL THEN 1 ELSE 0 END) as null_aqi,
                SUM(CASE WHEN "Arithmetic Mean" < 0 THEN 1 ELSE 0 END) as negative_pm25,
                SUM(CASE WHEN "AQI" > {self.config.aqi_max} THEN 1 ELSE 0 END) as extreme_aqi,
                ROUND(AVG("Arithmetic Mean"), 2) as avg_pm25,
                ROUND(MIN("Arithmetic Mean"), 2) as min_pm25,
                MAX("AQI") as max_aqi
            FROM '{self.data_path}'

            UNION ALL

            SELECT
                'NACHHER' as status,
                COUNT(*),
                SUM(CASE WHEN pm25_cleaned IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN aqi_cleaned IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN pm25_cleaned < 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN aqi_cleaned > {self.config.aqi_max} THEN 1 ELSE 0 END),
                ROUND(AVG(pm25_cleaned), 2),
                ROUND(MIN(pm25_cleaned), 2),
                MAX(aqi_cleaned)
            FROM cleaned_data
        """).fetchdf()

        print(comparison.to_string(index=False))

        # Erfolgs-Metriken berechnen
        problems_fixed = (
                self.stats_before['null_aqi'] +
                self.stats_before['negative_pm25'] +
                self.stats_before['extreme_aqi']
        )

        problems_remaining = (
                stats_after['null_aqi'] +
                stats_after['negative_pm25'] +
                stats_after['extreme_aqi']
        )

        success_rate = ((problems_fixed - problems_remaining) / problems_fixed * 100) if problems_fixed > 0 else 100

        print(f"\n🎯 ERFOLGS-METRIKEN:")
        print(f"   • Probleme identifiziert:  {problems_fixed:,}")
        print(f"   • Probleme behoben:        {problems_fixed - problems_remaining:,}")
        print(f"   • Verbleibende Probleme:   {problems_remaining:,}")
        print(f"   • Erfolgsrate:             {success_rate:.1f}%")

        if success_rate == 100:
            print("\n   🎉 PERFEKT! Alle Datenqualitätsprobleme erfolgreich behoben!")
        elif success_rate >= 95:
            print("\n   ✅ SEHR GUT! Fast alle Probleme behoben!")
        elif success_rate >= 80:
            print("\n   👍 GUT! Meiste Probleme behoben!")
        else:
            print("\n   ⚠️  WARNUNG: Einige Probleme bleiben bestehen!")

        return self.stats_before, stats_after

    # ========================================================================
    # PHASE 6: EXPORT
    # ========================================================================

    def export_cleaned_data(self, output_path: Optional[str] = None) -> Dict[str, Path]:
        """Exportiert bereinigte Daten in verschiedenen Formaten"""

        print("\n" + "=" * 80)
        print("💾 PHASE 6: EXPORT")
        print("=" * 80)

        if output_path is None:
            input_stem = self.data_path.stem
            output_dir = self.data_path.parent
        else:
            output_path = Path(output_path)
            input_stem = output_path.stem
            output_dir = output_path.parent

        exported_files = {}

        # Spalten-Auswahl (mit oder ohne Cleanup-Flags)
        if self.config.include_cleanup_flags:
            columns = "*"
        else:
            columns = """
                "State Name", "County Name", "Date Local", "Local Site Name",
                "Latitude", "Longitude", "Sample Duration", "Pollutant Standard",
                "Observation Percent", pm25_cleaned as "Arithmetic Mean", 
                aqi_cleaned as "AQI"
            """

        # Parquet Export
        if self.config.export_format in ["parquet", "both"]:
            parquet_path = output_dir / f"{input_stem}_cleaned.parquet"
            self.con.execute(f"""
                COPY (SELECT {columns} FROM cleaned_data)
                TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION 'ZSTD')
            """)
            exported_files['parquet'] = parquet_path

            # Größen-Info
            size_mb = parquet_path.stat().st_size / 1024 / 1024
            print(f"\n✅ Parquet Export: {parquet_path}")
            print(f"   • Größe: {size_mb:.2f} MB")
            print(f"   • Kompression: ZSTD")

        # CSV Export
        if self.config.export_format in ["csv", "both"]:
            csv_path = output_dir / f"{input_stem}_cleaned.csv"
            self.con.execute(f"""
                COPY (SELECT {columns} FROM cleaned_data)
                TO '{csv_path}' (FORMAT CSV, HEADER TRUE)
            """)
            exported_files['csv'] = csv_path

            # Größen-Info
            size_mb = csv_path.stat().st_size / 1024 / 1024
            print(f"\n✅ CSV Export: {csv_path}")
            print(f"   • Größe: {size_mb:.2f} MB")

        # Report generieren
        if self.config.verbose:
            self._generate_report(output_dir / f"{input_stem}_cleanup_report.txt")

        return exported_files

    def _generate_report(self, report_path: Path) -> None:
        """Generiert detaillierten Cleanup-Report"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EPA DATA CLEANUP REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.config.__dict__.items():
                f.write(f"  {key}: {value}\n")

            f.write("\n\nSTATISTICS BEFORE:\n")
            f.write("-" * 40 + "\n")
            if self.stats_before:
                for key, value in self.stats_before.items():
                    f.write(f"  {key}: {value}\n")

            f.write("\n\nSTATISTICS AFTER:\n")
            f.write("-" * 40 + "\n")
            if self.stats_after:
                for key, value in self.stats_after.items():
                    f.write(f"  {key}: {value}\n")

        print(f"\n📄 Report generiert: {report_path}")

    # ========================================================================
    # MAIN EXECUTION FLOW
    # ========================================================================

    def run_complete_pipeline(self) -> Dict[str, Path]:
        """Führt die komplette Pipeline aus"""

        print("\n" + "🎯 " * 30)
        print("🚀 EPA DATA CLEANUP PIPELINE - STARTING...")
        print("🎯 " * 30)

        try:
            # Phase 0: Setup
            self._connect_db()

            # Phase 1: Schema-Analyse
            self.analyze_schema()

            # Phase 2: Qualitäts-Statistiken (VORHER)
            self.calculate_quality_stats()

            # Phase 3: Problem-Beispiele
            self.show_problem_examples()

            # Phase 4: Data Cleanup
            self.apply_cleanup()

            # Phase 5: Validation
            self.validate_cleanup()

            # Phase 6: Export
            exported_files = self.export_cleaned_data()

            # Abschluss
            print("\n" + "🎉 " * 30)
            print("✅ PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
            print("🎉 " * 30)

            return exported_files

        except Exception as e:
            print(f"\n❌ PIPELINE FEHLER: {e}")
            raise
        finally:
            if self.con:
                self.con.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_run(
        data_path: str,
        export_format: str = "parquet",
        verbose: bool = True,
        show_examples: bool = True
) -> None:
    """
    Schnelle Ausführung mit Standard-Einstellungen

    Args:
        data_path: Pfad zur CSV-Datei
        export_format: "parquet", "csv", oder "both"
        verbose: Ausführliche Ausgabe?
        show_examples: Problem-Beispiele zeigen?
    """
    config = EPACleanupConfig(
        export_format=export_format,
        verbose=verbose,
        show_examples=show_examples
    )

    pipeline = EPADataQualityPipeline(data_path, config)
    pipeline.run_complete_pipeline()


def custom_run(data_path: str, config_dict: Dict) -> None:
    """
    Ausführung mit custom Konfiguration

    Args:
        data_path: Pfad zur CSV-Datei
        config_dict: Dictionary mit Config-Parametern
    """
    config = EPACleanupConfig(**config_dict)
    pipeline = EPADataQualityPipeline(data_path, config)
    pipeline.run_complete_pipeline()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Standard-Pfad (anpassen!)
    DATA_PATH = r'C:\Users\sebas\PycharmProjects\Big_Data_Umwelbehörde\Data\daily_88101_2024.csv'

    # Option 1: Quick Run mit Defaults
    try:
        quick_run(DATA_PATH, export_format="both")
    except FileNotFoundError:
        print("\n⚠️  Bitte passe den DATA_PATH an deine Datei an!")
        print("   Beispiel: DATA_PATH = '/path/to/ad_viz_plotval_data.csv'")
        sys.exit(1)

    # Option 2: Custom Configuration (auskommentiert)
    """
    custom_config = {
        "null_aqi_strategy": "calculate",
        "fix_negative_pm25": True,
        "cap_extreme_aqi": True,
        "show_examples": True,
        "export_format": "parquet",
        "verbose": True,
        "example_limit": 5,
        "pm25_extreme_threshold": 200.0
    }
    custom_run(DATA_PATH, custom_config)
    """

    # Option 3: Minimal Configuration (auskommentiert)
    """
    minimal_config = EPACleanupConfig(
        verbose=False,
        show_examples=False,
        export_format="parquet"
    )
    pipeline = EPADataQualityPipeline(DATA_PATH, minimal_config)
    pipeline.run_complete_pipeline()
    """