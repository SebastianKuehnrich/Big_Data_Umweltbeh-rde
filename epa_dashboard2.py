# =============================================================================
# EPA AIR QUALITY DASHBOARD v2.0 - PROFESSIONAL EDITION
# Author: Sebastian Kühnrich
# Updated for pre-cleaned data structure
# =============================================================================

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from io import BytesIO
import hashlib
import json


# =============================================================================
# CONFIGURATION & MODELS
# =============================================================================

@dataclass
class FilterState:
    """Filter state for dashboard."""
    selected_states: List[str] = None
    date_from: date = None
    date_to: date = None
    comparison_mode: bool = False
    show_anomalies: bool = False

    def __post_init__(self):
        if self.selected_states is None:
            self.selected_states = ["All States"]
        if self.date_from is None:
            self.date_from = date(2024, 1, 1)
        if self.date_to is None:
            self.date_to = date(2024, 12, 31)


@dataclass
class ChartConfig:
    """Configuration for chart rendering."""
    height: int = 400
    show_legend: bool = True
    color_scheme: str = "viridis"
    animation: bool = True


@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment."""
    total_records: int
    null_count: int
    negative_count: int
    outlier_count: int
    completeness_score: float

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        if self.total_records == 0:
            return 0
        return max(0, 100 - (
                (self.null_count / self.total_records * 20) +
                (self.negative_count / self.total_records * 20) +
                (self.outlier_count / self.total_records * 10)
        ))


class AQICategory(Enum):
    """AQI Categories based on EPA standards."""
    GOOD = "Good (0-50)"
    MODERATE = "Moderate (51-100)"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive (101-150)"
    UNHEALTHY = "Unhealthy (151-200)"
    VERY_UNHEALTHY = "Very Unhealthy (201-300)"
    HAZARDOUS = "Hazardous (301-500)"


# =============================================================================
# COLUMN CONFIGURATION FOR PRE-CLEANED DATA
# =============================================================================

# Actual column names in the cleaned dataset
COLUMN_NAMES = {
    'PM25': 'pm25_cleaned',
    'AQI': 'aqi_cleaned',
    'PM25_ORIGINAL': 'pm25_original',
    'AQI_ORIGINAL': 'aqi_original',
    'STATE': 'State Name',
    'COUNTY': 'County Name',
    'DATE': 'Date Local',
    'SITE': 'Local Site Name',
    'LAT': 'Latitude',
    'LON': 'Longitude',
    'DURATION': 'Sample Duration',
    'STANDARD': 'Pollutant Standard',
    'PERCENT': 'Observation Percent',
    'QUALITY_FLAG': 'data_quality_flag',
    'CLEANUP': 'cleanup_action'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class QueryBuilder:
    """SQL query builder with validation."""

    @staticmethod
    def build_where_clause(
            states: List[str],
            date_from: date,
            date_to: date,
            additional_conditions: Optional[List[str]] = None
    ) -> str:
        """Build optimized WHERE clause."""
        conditions = []

        if states and "All States" not in states:
            state_list = "', '".join(states)
            conditions.append(f'"{COLUMN_NAMES["STATE"]}" IN (\'{state_list}\')')

        conditions.append(f'"{COLUMN_NAMES["DATE"]}" >= \'{date_from}\'')
        conditions.append(f'"{COLUMN_NAMES["DATE"]}" <= \'{date_to}\'')

        if additional_conditions:
            conditions.extend(additional_conditions)

        return "WHERE " + " AND ".join(conditions) if conditions else ""


class AnomalyDetector:
    """Detect anomalies in air quality data."""

    @staticmethod
    def detect_anomalies(
            df: pd.DataFrame,
            column: str,
            method: str = 'zscore',
            threshold: float = 3.0
    ) -> pd.DataFrame:
        """Detect anomalies using various methods."""
        df = df.copy()

        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            df['is_anomaly'] = False
            df.loc[df[column].notna(), 'is_anomaly'] = z_scores > threshold
            df['anomaly_score'] = z_scores

        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df['is_anomaly'] = (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))

        return df


def format_number(num: float, precision: int = 2) -> str:
    """Format number with thousands separator."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.{precision}f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def calculate_trend(data: pd.Series) -> Tuple[float, str]:
    """Calculate trend direction and magnitude."""
    if len(data) < 2:
        return 0.0, "stable"

    x = np.arange(len(data))
    slope, _ = np.polyfit(x, data.values, 1)
    trend_pct = (slope / data.mean()) * 100 if data.mean() != 0 else 0

    if abs(trend_pct) < 1:
        return trend_pct, "stable"
    elif trend_pct > 0:
        return trend_pct, "increasing"
    else:
        return trend_pct, "decreasing"


# =============================================================================
# DATA LOADER FOR PRE-CLEANED DATA
# =============================================================================

class DataLoader:
    """Handle all data operations for the dashboard."""

    def __init__(self, data_path: str):
        """Initialize data loader."""
        self.data_path = data_path
        self.query_builder = QueryBuilder()
        self.connection = None
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize DuckDB connection."""
        try:
            self.connection = duckdb.connect(':memory:')
        except Exception as e:
            st.error(f"Failed to initialize DuckDB: {e}")
            raise

    @st.cache_data(ttl=3600)
    def get_kpi_metrics(_self) -> Dict[str, Any]:
        """Get KPI metrics from pre-cleaned data."""
        query = f"""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT "{COLUMN_NAMES['STATE']}") as unique_states,
            ROUND(AVG({COLUMN_NAMES['PM25']}), 2) as avg_pm25,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {COLUMN_NAMES['PM25']}), 2) as median_pm25,
            ROUND(STDDEV({COLUMN_NAMES['PM25']}), 2) as std_pm25,
            ROUND(AVG({COLUMN_NAMES['AQI']}), 1) as avg_aqi,
            SUM(CASE WHEN {COLUMN_NAMES['PM25_ORIGINAL']} < 0 THEN 1 ELSE 0 END) as negative_fixed,
            SUM(CASE WHEN {COLUMN_NAMES['AQI_ORIGINAL']} IS NULL THEN 1 ELSE 0 END) as null_aqi_fixed,
            SUM(CASE WHEN {COLUMN_NAMES['AQI']} > 500 THEN 1 ELSE 0 END) as extreme_aqi_capped,
            MIN("{COLUMN_NAMES['DATE']}") as date_start,
            MAX("{COLUMN_NAMES['DATE']}") as date_end,
            SUM(CASE WHEN {COLUMN_NAMES['QUALITY_FLAG']} = 'cleaned' THEN 1 ELSE 0 END) as cleaned_records
        FROM '{_self.data_path}'
        """

        try:
            result = _self.connection.execute(query).fetchdf()
            return result.to_dict('records')[0]
        except Exception as e:
            st.error(f"Failed to get KPI metrics: {e}")
            return {}

    @st.cache_data(ttl=3600)
    def get_monthly_trends(_self, filter_state: FilterState) -> pd.DataFrame:
        """Get monthly trends from pre-cleaned data."""
        where_clause = _self.query_builder.build_where_clause(
            filter_state.selected_states,
            filter_state.date_from,
            filter_state.date_to
        )

        query = f"""
        SELECT
            STRFTIME("{COLUMN_NAMES['DATE']}"::DATE, '%Y-%m') as month,
            {'"' + COLUMN_NAMES['STATE'] + '",' if filter_state.comparison_mode else ''}
            ROUND(AVG({COLUMN_NAMES['PM25']}), 2) as avg_pm25,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {COLUMN_NAMES['PM25']}), 2) as q1_pm25,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {COLUMN_NAMES['PM25']}), 2) as median_pm25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {COLUMN_NAMES['PM25']}), 2) as q3_pm25,
            ROUND(AVG({COLUMN_NAMES['AQI']}), 1) as avg_aqi,
            SUM(CASE WHEN {COLUMN_NAMES['AQI']} > 100 THEN 1 ELSE 0 END) as unhealthy_days,
            COUNT(*) as measurements,
            ROUND(STDDEV({COLUMN_NAMES['PM25']}), 2) as std_pm25
        FROM '{_self.data_path}'
        {where_clause}
        GROUP BY month {', "' + COLUMN_NAMES['STATE'] + '"' if filter_state.comparison_mode else ''}
        ORDER BY month
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get monthly trends: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_state_comparison(_self, filter_state: FilterState) -> pd.DataFrame:
        """Get state-by-state comparison from pre-cleaned data."""
        where_clause = _self.query_builder.build_where_clause(
            filter_state.selected_states,
            filter_state.date_from,
            filter_state.date_to
        )

        query = f"""
        SELECT
            "{COLUMN_NAMES['STATE']}" as "State Name",
            COUNT(DISTINCT "{COLUMN_NAMES['COUNTY']}") as counties,
            COUNT(*) as measurements,
            ROUND(AVG({COLUMN_NAMES['PM25']}), 2) as avg_pm25,
            ROUND(MAX({COLUMN_NAMES['PM25']}), 2) as max_pm25,
            ROUND(MIN({COLUMN_NAMES['PM25']}), 2) as min_pm25,
            ROUND(AVG({COLUMN_NAMES['AQI']}), 1) as avg_aqi,
            SUM(CASE WHEN {COLUMN_NAMES['AQI']} <= 50 THEN 1 ELSE 0 END) as good_days,
            SUM(CASE WHEN {COLUMN_NAMES['AQI']} > 50 AND {COLUMN_NAMES['AQI']} <= 100 THEN 1 ELSE 0 END) as moderate_days,
            SUM(CASE WHEN {COLUMN_NAMES['AQI']} > 100 THEN 1 ELSE 0 END) as unhealthy_days,
            ROUND(100.0 * SUM(CASE WHEN {COLUMN_NAMES['AQI']} <= 50 THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_good_days
        FROM '{_self.data_path}'
        {where_clause}
        GROUP BY "{COLUMN_NAMES['STATE']}"
        ORDER BY avg_pm25 DESC
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get state comparison: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_data_quality_metrics(_self) -> DataQualityMetrics:
        """Analyze data quality from pre-cleaned data."""
        query = f"""
        SELECT
            COUNT(*) as total_records,
            SUM(CASE WHEN {COLUMN_NAMES['PM25']} IS NULL THEN 1 ELSE 0 END) as null_count,
            SUM(CASE WHEN {COLUMN_NAMES['PM25_ORIGINAL']} < 0 THEN 1 ELSE 0 END) as negative_count,
            SUM(CASE WHEN ABS({COLUMN_NAMES['PM25']} - avg_val) > 3 * std_val THEN 1 ELSE 0 END) as outlier_count
        FROM '{_self.data_path}',
        (SELECT AVG({COLUMN_NAMES['PM25']}) as avg_val, STDDEV({COLUMN_NAMES['PM25']}) as std_val FROM '{_self.data_path}')
        """

        try:
            result = _self.connection.execute(query).fetchdf().iloc[0]
            completeness = 1 - (result['null_count'] / result['total_records']) if result['total_records'] > 0 else 0

            return DataQualityMetrics(
                total_records=int(result['total_records']),
                null_count=int(result['null_count']),
                negative_count=int(result['negative_count']),
                outlier_count=int(result['outlier_count']),
                completeness_score=completeness * 100
            )
        except Exception as e:
            st.error(f"Failed to get data quality metrics: {e}")
            return DataQualityMetrics(0, 0, 0, 0, 0)

    @st.cache_data(ttl=3600)
    def get_states_list(_self) -> List[str]:
        """Get list of all available states."""
        query = f"""
        SELECT DISTINCT "{COLUMN_NAMES['STATE']}"
        FROM '{_self.data_path}'
        WHERE "{COLUMN_NAMES['STATE']}" IS NOT NULL
        ORDER BY "{COLUMN_NAMES['STATE']}"
        """

        try:
            result = _self.connection.execute(query).fetchdf()
            return result[COLUMN_NAMES['STATE']].tolist()
        except Exception as e:
            st.error(f"Failed to get states list: {e}")
            return []

    def get_raw_data(self, filter_state: FilterState, limit: int = 1000) -> pd.DataFrame:
        """Get raw data with filters applied."""
        where_clause = self.query_builder.build_where_clause(
            filter_state.selected_states,
            filter_state.date_from,
            filter_state.date_to
        )

        query = f"""
        SELECT *
        FROM '{self.data_path}'
        {where_clause}
        LIMIT {limit}
        """

        try:
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get raw data: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_aqi_distribution(_self, where_clause: str) -> pd.DataFrame:
        """Get AQI category distribution from pre-cleaned data."""
        query = f"""
        SELECT
            CASE
                WHEN {COLUMN_NAMES['AQI']} <= 50 THEN '1. Good (0-50)'
                WHEN {COLUMN_NAMES['AQI']} <= 100 THEN '2. Moderate (51-100)'
                WHEN {COLUMN_NAMES['AQI']} <= 150 THEN '3. Unhealthy Sensitive (101-150)'
                WHEN {COLUMN_NAMES['AQI']} <= 200 THEN '4. Unhealthy (151-200)'
                WHEN {COLUMN_NAMES['AQI']} <= 300 THEN '5. Very Unhealthy (201-300)'
                ELSE '6. Hazardous (301-500)'
            END as category,
            COUNT(*) as count
        FROM '{_self.data_path}'
        {where_clause}
        GROUP BY category
        ORDER BY category
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except:
            return pd.DataFrame()

    # =========================================================================
    # ADVANCED ANALYTICS QUERIES (Homework Tag14)
    # =========================================================================

    @st.cache_data(ttl=3600)
    def get_alarm_days(_self, filter_state: FilterState) -> pd.DataFrame:
        """Query 1: Find days with >100% PM2.5 increase using LAG window function."""
        where_conditions = []
        if filter_state.selected_states and "All States" not in filter_state.selected_states:
            state_list = "', '".join(filter_state.selected_states)
            where_conditions.append(f'"{COLUMN_NAMES["STATE"]}" IN (\'{state_list}\')')

        base_where = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

        query = f"""
        WITH daily_avg AS (
            SELECT
                "{COLUMN_NAMES['STATE']}" as state_name,
                "{COLUMN_NAMES['DATE']}"::DATE as date_local,
                ROUND(AVG(GREATEST(COALESCE({COLUMN_NAMES['PM25']}, 0), 0)), 2) as pm25
            FROM '{_self.data_path}'
            {base_where}
            GROUP BY "{COLUMN_NAMES['STATE']}", "{COLUMN_NAMES['DATE']}"
        ),
        with_yesterday AS (
            SELECT
                state_name,
                date_local,
                pm25 as heute,
                LAG(pm25) OVER (
                    PARTITION BY state_name
                    ORDER BY date_local
                ) as gestern
            FROM daily_avg
        )
        SELECT
            state_name as "State Name",
            date_local as "Date Local",
            heute,
            gestern,
            ROUND((heute - gestern) / NULLIF(gestern, 0) * 100, 1) as prozent_change,
            CASE 
                WHEN (heute - gestern) / NULLIF(gestern, 0) * 100 > 200 THEN 'KRITISCH'
                WHEN (heute - gestern) / NULLIF(gestern, 0) * 100 > 100 THEN 'WARNUNG'
                ELSE 'Normal' 
            END as alarm_level
        FROM with_yesterday
        WHERE gestern > 0
          AND gestern IS NOT NULL
          AND (heute - gestern) / NULLIF(gestern, 0) * 100 > 100
        ORDER BY prozent_change DESC
        LIMIT 100
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get alarm days: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_state_quarterly_trend(_self) -> pd.DataFrame:
        """Query 2: Compare Q1 vs Q4 to find improvement/worsening states."""
        query = f"""
        WITH quarterly_data AS (
            SELECT
                "{COLUMN_NAMES['STATE']}" as state_name,
                CASE 
                    WHEN MONTH("{COLUMN_NAMES['DATE']}"::DATE) <= 3 THEN 'Q1'
                    WHEN MONTH("{COLUMN_NAMES['DATE']}"::DATE) <= 6 THEN 'Q2'
                    WHEN MONTH("{COLUMN_NAMES['DATE']}"::DATE) <= 9 THEN 'Q3'
                    ELSE 'Q4'
                END as quartal,
                ROUND(AVG(GREATEST(COALESCE({COLUMN_NAMES['PM25']}, 0), 0)), 2) as avg_pm25
            FROM '{_self.data_path}'
            GROUP BY "{COLUMN_NAMES['STATE']}", quartal
        ),
        q1_q4_comparison AS (
            SELECT
                state_name,
                MAX(CASE WHEN quartal = 'Q1' THEN avg_pm25 END) as q1_avg,
                MAX(CASE WHEN quartal = 'Q4' THEN avg_pm25 END) as q4_avg
            FROM quarterly_data
            GROUP BY state_name
            HAVING q1_avg IS NOT NULL AND q4_avg IS NOT NULL
        )
        SELECT
            state_name as "State Name",
            q1_avg,
            q4_avg,
            ROUND(q4_avg - q1_avg, 2) as diff,
            ROUND((q4_avg - q1_avg) / NULLIF(q1_avg, 0) * 100, 1) as prozent_change,
            RANK() OVER (ORDER BY (q4_avg - q1_avg) ASC) as improvement_rank,
            CASE 
                WHEN q4_avg < q1_avg - 1 THEN 'Verbessert'
                WHEN q4_avg > q1_avg + 1 THEN 'Verschlechtert'
                ELSE 'Stabil'
            END as kategorie
        FROM q1_q4_comparison
        ORDER BY diff ASC
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get state trend: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_worst_days_per_state(_self, filter_state: FilterState) -> pd.DataFrame:
        """Query 3: Top 3 worst days per state using ROW_NUMBER."""
        where_conditions = []
        if filter_state.selected_states and "All States" not in filter_state.selected_states:
            state_list = "', '".join(filter_state.selected_states)
            where_conditions.append(f'"{COLUMN_NAMES["STATE"]}" IN (\'{state_list}\')')

        base_where = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

        query = f"""
        WITH daily_ranked AS (
            SELECT
                "{COLUMN_NAMES['STATE']}" as state_name,
                "{COLUMN_NAMES['DATE']}"::DATE as date_local,
                ROUND(AVG(GREATEST(COALESCE({COLUMN_NAMES['PM25']}, 0), 0)), 2) as pm25,
                ROW_NUMBER() OVER (
                    PARTITION BY "{COLUMN_NAMES['STATE']}" 
                    ORDER BY AVG(GREATEST(COALESCE({COLUMN_NAMES['PM25']}, 0), 0)) DESC
                ) as rang,
                LAG(ROUND(AVG(GREATEST(COALESCE({COLUMN_NAMES['PM25']}, 0), 0)), 2)) OVER (
                    PARTITION BY "{COLUMN_NAMES['STATE']}" 
                    ORDER BY "{COLUMN_NAMES['DATE']}"::DATE
                ) as vortag
            FROM '{_self.data_path}'
            {base_where}
            GROUP BY "{COLUMN_NAMES['STATE']}", "{COLUMN_NAMES['DATE']}"
        ),
        with_us_avg AS (
            SELECT 
                *,
                (SELECT ROUND(AVG(pm25), 2) FROM daily_ranked) as us_avg
            FROM daily_ranked
        )
        SELECT
            state_name as "State Name",
            date_local as "Date Local",
            pm25,
            rang,
            vortag,
            ROUND(pm25 - COALESCE(vortag, pm25), 2) as sprung,
            ROUND(pm25 - us_avg, 2) as ueber_durchschnitt,
            CASE WHEN pm25 - COALESCE(vortag, pm25) > 20 THEN 'Plötzlich' ELSE 'Normal' END as sprung_typ
        FROM with_us_avg
        WHERE rang <= 3
        ORDER BY state_name, rang
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get worst days: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_rolling_average(_self, state: str) -> pd.DataFrame:
        """Query 4: 7-day rolling average with trend indicator."""
        query = f"""
        WITH daily_avg AS (
            SELECT
                "{COLUMN_NAMES['DATE']}"::DATE as date_local,
                ROUND(AVG(GREATEST(COALESCE({COLUMN_NAMES['PM25']}, 0), 0)), 2) as pm25
            FROM '{_self.data_path}'
            WHERE "{COLUMN_NAMES['STATE']}" = '{state}'
            GROUP BY "{COLUMN_NAMES['DATE']}"
        )
        SELECT
            date_local as "Date Local",
            pm25 as daily_pm25,
            ROUND(AVG(pm25) OVER (
                ORDER BY date_local
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 2) as rolling_7day,
            CASE 
                WHEN pm25 > AVG(pm25) OVER (ORDER BY date_local ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) 
                THEN 'Über Trend' 
                ELSE 'Unter Trend' 
            END as trend_position
        FROM daily_avg
        ORDER BY date_local
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get rolling average: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_weekday_analysis(_self) -> pd.DataFrame:
        """Bonus Challenge 1: Weekday air quality analysis."""
        query = f"""
        SELECT
            DAYOFWEEK("{COLUMN_NAMES['DATE']}"::DATE) as day_num,
            CASE DAYOFWEEK("{COLUMN_NAMES['DATE']}"::DATE)
                WHEN 0 THEN 'Sonntag'
                WHEN 1 THEN 'Montag'
                WHEN 2 THEN 'Dienstag'
                WHEN 3 THEN 'Mittwoch'
                WHEN 4 THEN 'Donnerstag'
                WHEN 5 THEN 'Freitag'
                WHEN 6 THEN 'Samstag'
            END as wochentag,
            COUNT(*) as messungen,
            ROUND(AVG({COLUMN_NAMES['PM25']}), 2) as avg_pm25,
            ROUND(AVG({COLUMN_NAMES['AQI']}), 1) as avg_aqi,
            ROUND(STDDEV({COLUMN_NAMES['PM25']}), 2) as std_pm25
        FROM '{_self.data_path}'
        GROUP BY day_num, wochentag
        ORDER BY day_num
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get weekday analysis: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_good_air_streaks(_self, top_n: int = 10) -> pd.DataFrame:
        """Bonus Challenge 2: Find longest streaks of good air quality (PM2.5 < 12)."""
        query = f"""
        WITH daily_status AS (
            SELECT
                "{COLUMN_NAMES['STATE']}" as state_name,
                "{COLUMN_NAMES['DATE']}"::DATE as date_local,
                ROUND(AVG({COLUMN_NAMES['PM25']}), 2) as pm25,
                CASE WHEN AVG({COLUMN_NAMES['PM25']}) < 12 THEN 1 ELSE 0 END as is_good
            FROM '{_self.data_path}'
            GROUP BY "{COLUMN_NAMES['STATE']}", "{COLUMN_NAMES['DATE']}"
        ),
        streak_groups AS (
            SELECT
                state_name,
                date_local,
                pm25,
                is_good,
                date_local - INTERVAL (ROW_NUMBER() OVER (
                    PARTITION BY state_name, is_good 
                    ORDER BY date_local
                )) DAY as streak_group
            FROM daily_status
        ),
        streak_lengths AS (
            SELECT
                state_name,
                MIN(date_local) as streak_start,
                MAX(date_local) as streak_end,
                COUNT(*) as streak_length,
                is_good
            FROM streak_groups
            WHERE is_good = 1
            GROUP BY state_name, streak_group, is_good
        )
        SELECT
            state_name as "State Name",
            streak_start as "Streak Start",
            streak_end as "Streak End",
            streak_length as "Tage",
            RANK() OVER (PARTITION BY state_name ORDER BY streak_length DESC) as state_rank
        FROM streak_lengths
        QUALIFY state_rank = 1
        ORDER BY streak_length DESC
        LIMIT {top_n}
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get good air streaks: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_anomaly_scores(_self, filter_state: FilterState) -> pd.DataFrame:
        """Bonus Challenge 3: Calculate z-score anomaly for each day."""
        where_conditions = []
        if filter_state.selected_states and "All States" not in filter_state.selected_states:
            state_list = "', '".join(filter_state.selected_states)
            where_conditions.append(f'"{COLUMN_NAMES["STATE"]}" IN (\'{state_list}\')')

        base_where = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

        query = f"""
        WITH daily_data AS (
            SELECT
                "{COLUMN_NAMES['STATE']}" as state_name,
                "{COLUMN_NAMES['DATE']}"::DATE as date_local,
                ROUND(AVG({COLUMN_NAMES['PM25']}), 2) as pm25
            FROM '{_self.data_path}'
            {base_where}
            GROUP BY "{COLUMN_NAMES['STATE']}", "{COLUMN_NAMES['DATE']}"
        ),
        with_stats AS (
            SELECT
                state_name,
                date_local,
                pm25,
                AVG(pm25) OVER (PARTITION BY state_name) as state_avg,
                STDDEV(pm25) OVER (PARTITION BY state_name) as state_stddev
            FROM daily_data
        )
        SELECT
            state_name as "State Name",
            date_local as "Date Local",
            pm25,
            ROUND(state_avg, 2) as state_avg,
            ROUND(state_stddev, 2) as state_stddev,
            ROUND((pm25 - state_avg) / NULLIF(state_stddev, 0), 2) as z_score,
            CASE 
                WHEN ABS((pm25 - state_avg) / NULLIF(state_stddev, 0)) > 3 THEN 'Extreme Anomalie'
                WHEN ABS((pm25 - state_avg) / NULLIF(state_stddev, 0)) > 2 THEN 'Starke Anomalie'
                WHEN ABS((pm25 - state_avg) / NULLIF(state_stddev, 0)) > 1.5 THEN 'Leichte Anomalie'
                ELSE 'Normal'
            END as anomaly_level
        FROM with_stats
        WHERE state_stddev > 0
        ORDER BY ABS((pm25 - state_avg) / NULLIF(state_stddev, 0)) DESC
        LIMIT 100
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get anomaly scores: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_site_analysis(_self, filter_state: FilterState) -> pd.DataFrame:
        """Get analysis by monitoring sites."""
        where_clause = _self.query_builder.build_where_clause(
            filter_state.selected_states,
            filter_state.date_from,
            filter_state.date_to
        )

        query = f"""
        SELECT
            "{COLUMN_NAMES['SITE']}" as site_name,
            "{COLUMN_NAMES['STATE']}" as state,
            "{COLUMN_NAMES['COUNTY']}" as county,
            ROUND(AVG({COLUMN_NAMES['LAT']}), 4) as latitude,
            ROUND(AVG({COLUMN_NAMES['LON']}), 4) as longitude,
            COUNT(*) as measurements,
            ROUND(AVG({COLUMN_NAMES['PM25']}), 2) as avg_pm25,
            ROUND(AVG({COLUMN_NAMES['AQI']}), 1) as avg_aqi,
            ROUND(AVG("{COLUMN_NAMES['PERCENT']}"), 1) as avg_observation_pct
        FROM '{_self.data_path}'
        {where_clause}
        GROUP BY "{COLUMN_NAMES['SITE']}", "{COLUMN_NAMES['STATE']}", "{COLUMN_NAMES['COUNTY']}"
        ORDER BY avg_pm25 DESC
        LIMIT 20
        """

        try:
            return _self.connection.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Failed to get site analysis: {e}")
            return pd.DataFrame()


# =============================================================================
# VISUALIZATIONS
# =============================================================================

class Visualizations:
    """Create interactive visualizations for dashboard."""

    def __init__(self, chart_config: Optional[ChartConfig] = None):
        """Initialize visualization settings."""
        self.config = chart_config or ChartConfig()
        self.colors = {
            'good': '#00E400',
            'moderate': '#FFFF00',
            'unhealthy_sensitive': '#FF7E00',
            'unhealthy': '#FF0000',
            'very_unhealthy': '#8F3F97',
            'hazardous': '#7E0023'
        }
        self.thresholds = {
            'pm25': {
                'safe': 12.0,
                'moderate': 35.4,
                'unhealthy': 55.4,
                'very_unhealthy': 150.4,
                'hazardous': 250.4
            }
        }

    def create_trend_chart(self, df: pd.DataFrame, comparison_mode: bool = False) -> go.Figure:
        """Create interactive trend chart with multiple metrics."""
        if comparison_mode and 'State Name' in df.columns:
            fig = px.line(
                df,
                x='month',
                y='avg_pm25',
                color='State Name',
                title='PM2.5 Trends by State',
                labels={'avg_pm25': 'PM2.5 (µg/m³)', 'month': 'Month'},
                height=self.config.height,
                markers=True
            )
        else:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['month'],
                y=df['avg_pm25'],
                mode='lines+markers',
                name='Average PM2.5',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8)
            ))

            if 'q1_pm25' in df.columns and 'q3_pm25' in df.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([df['month'], df['month'][::-1]]),
                    y=pd.concat([df['q3_pm25'], df['q1_pm25'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='IQR Range',
                    hoverinfo='skip'
                ))

            fig.update_layout(
                title='Air Quality Trend Analysis',
                xaxis_title='Month',
                yaxis_title='PM2.5 (µg/m³)',
                height=self.config.height,
                hovermode='x unified',
                showlegend=True
            )

        # Add threshold lines
        for name, threshold in self.thresholds['pm25'].items():
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=self.colors.get(name.replace('_', ''), 'gray'),
                annotation_text=name.replace('_', ' ').title(),
                annotation_position="right",
                opacity=0.5
            )

        return fig

    def create_aqi_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create AQI distribution donut chart."""
        if df.empty:
            return go.Figure()

        colors_map = {
            '1. Good (0-50)': self.colors['good'],
            '2. Moderate (51-100)': self.colors['moderate'],
            '3. Unhealthy Sensitive (101-150)': self.colors['unhealthy_sensitive'],
            '4. Unhealthy (151-200)': self.colors['unhealthy'],
            '5. Very Unhealthy (201-300)': self.colors['very_unhealthy'],
            '6. Hazardous (301-500)': self.colors['hazardous']
        }

        fig = go.Figure(data=[go.Pie(
            labels=df['category'],
            values=df['count'],
            hole=0.4,
            marker=dict(colors=[colors_map.get(cat, 'gray') for cat in df['category']]),
            textposition='auto',
            textinfo='label+percent'
        )])

        fig.update_layout(
            title='AQI Category Distribution',
            height=self.config.height,
            annotations=[dict(
                text='AQI',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )]
        )

        return fig

    def create_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap for state comparison."""
        if df.empty:
            return go.Figure()

        metrics = ['avg_pm25', 'avg_aqi', 'pct_good_days']
        metric_names = ['Avg PM2.5', 'Avg AQI', 'Good Days %']

        heatmap_data = []
        for metric in metrics:
            if metric in df.columns:
                normalized = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                heatmap_data.append(normalized.values)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=df['State Name'].values[:10],  # Top 10 states
            y=metric_names[:len(heatmap_data)],
            colorscale='RdYlGn_r',
            showscale=True,
            hoverongaps=False,
            hovertemplate='State: %{x}<br>Metric: %{y}<br>Value: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title='State Air Quality Comparison Heatmap (Top 10)',
            height=self.config.height,
            xaxis_title='State',
            yaxis_title='Metric'
        )

        return fig

    def create_gauge_chart(self, value: float, title: str, max_value: float = 500) -> go.Figure:
        """Create gauge chart for current AQI."""
        if value <= 50:
            color = self.colors['good']
        elif value <= 100:
            color = self.colors['moderate']
        elif value <= 150:
            color = self.colors['unhealthy_sensitive']
        elif value <= 200:
            color = self.colors['unhealthy']
        elif value <= 300:
            color = self.colors['very_unhealthy']
        else:
            color = self.colors['hazardous']

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [None, max_value], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'lightgray'},
                    {'range': [50, 100], 'color': 'gray'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))

        fig.update_layout(height=250)
        return fig

    def create_site_map(self, df: pd.DataFrame) -> go.Figure:
        """Create map visualization of monitoring sites."""
        if df.empty or 'latitude' not in df.columns:
            return go.Figure()

        fig = px.scatter_mapbox(
            df,
            lat='latitude',
            lon='longitude',
            size='avg_pm25',
            color='avg_aqi',
            hover_name='site_name',
            hover_data=['state', 'county', 'avg_pm25', 'avg_aqi'],
            color_continuous_scale='RdYlGn_r',
            size_max=15,
            zoom=3,
            height=self.config.height,
            title='Air Quality Monitoring Sites'
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )

        return fig


# =============================================================================
# MAIN DASHBOARD APPLICATION
# =============================================================================

class EPADashboard:
    """Main dashboard application."""

    def __init__(self, data_path: str):
        """Initialize dashboard components."""
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.visualizations = Visualizations()
        self.anomaly_detector = AnomalyDetector()
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = FilterState()

        if 'selected_tab' not in st.session_state:
            st.session_state.selected_tab = "Overview"

    def run(self) -> None:
        """Run the dashboard application."""
        # Configure page
        st.set_page_config(
            page_title="EPA Air Quality Dashboard v2.1 - Advanced Analytics",
            page_icon="🌬️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply custom CSS
        self._apply_custom_css()

        # Render components
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
        self._render_footer()

    def _apply_custom_css(self) -> None:
        """Apply custom CSS styling with improved readability."""
        st.markdown("""
        <style>
        /* Main container */
        .main { 
            padding: 0rem 1rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        /* Improved Tab styling */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 8px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            padding: 10px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.4) 0%, rgba(118, 75, 162, 0.4) 100%);
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
        }

        /* Tab Panel Background */
        .stTabs [data-baseweb="tab-panel"] {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 15px;
            padding: 20px;
            margin-top: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Metric cards enhancement */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }

        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        }

        /* Headers with gradient */
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        h2, h3 {
            color: #ffffff !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        /* Buttons enhancement */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 20px 0 rgba(31, 38, 135, 0.4);
        }

        /* Dataframe styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Expander enhancement */
        .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Sidebar enhancement */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        }

        /* Select boxes and inputs */
        .stSelectbox > div > div,
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_header(self) -> None:
        """Render dashboard header."""
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title("🌬️ EPA Air Quality Dashboard v2.1")
            st.caption(f"Professional Edition with Advanced Analytics | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    def _render_sidebar(self) -> None:
        """Render sidebar with filters."""
        st.sidebar.header("🔎 Filters & Settings")

        # Data info
        with st.sidebar.expander("📊 Data Structure", expanded=False):
            st.success("✅ Using pre-cleaned data")
            st.info(f"""
            **Available columns:**
            - PM2.5: `{COLUMN_NAMES['PM25']}`
            - AQI: `{COLUMN_NAMES['AQI']}`
            - State: `{COLUMN_NAMES['STATE']}`
            - Date: `{COLUMN_NAMES['DATE']}`
            """)

        # Get available states
        states_list = self.data_loader.get_states_list()
        all_states_option = ["All States"] + states_list

        # State selection
        if st.sidebar.checkbox("Enable Multi-State Comparison", key="comparison_mode"):
            st.session_state.filter_state.comparison_mode = True
            selected_states = st.sidebar.multiselect(
                "Select States to Compare",
                options=states_list,
                default=states_list[:3] if len(states_list) >= 3 else states_list,
                key="state_multi"
            )
            st.session_state.filter_state.selected_states = selected_states
        else:
            st.session_state.filter_state.comparison_mode = False
            selected_state = st.sidebar.selectbox(
                "Select State",
                options=all_states_option,
                index=0,
                key="state_single"
            )
            st.session_state.filter_state.selected_states = [selected_state]

        # Date range
        st.sidebar.subheader("📅 Date Range")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            date_from = st.date_input(
                "From",
                value=st.session_state.filter_state.date_from,
                min_value=date(2024, 1, 1),
                max_value=date(2024, 12, 31),
                key="date_from"
            )

        with col2:
            date_to = st.date_input(
                "To",
                value=st.session_state.filter_state.date_to,
                min_value=date(2024, 1, 1),
                max_value=date(2024, 12, 31),
                key="date_to"
            )

        st.session_state.filter_state.date_from = date_from
        st.session_state.filter_state.date_to = date_to

        # Advanced options
        with st.sidebar.expander("⚙️ Advanced Settings"):
            st.session_state.filter_state.show_anomalies = st.checkbox(
                "Highlight Anomalies",
                value=st.session_state.filter_state.show_anomalies,
                key="show_anomalies"
            )

            chart_height = st.slider(
                "Chart Height",
                min_value=300,
                max_value=600,
                value=400,
                step=50,
                key="chart_height"
            )
            self.visualizations.config.height = chart_height

        # Info section
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **📊 Dashboard Features:**
        - Pre-cleaned data processing
        - Interactive visualizations
        - Multi-state comparison
        - Anomaly detection
        - Site location mapping
        - Data quality monitoring
        
        **🔥 NEW - Advanced Analytics:**
        - LAG Window Functions
        - ROW_NUMBER Rankings
        - Rolling Averages
        - Quartals-Vergleiche
        - Z-Score Anomalien
        """)

    def _render_main_content(self) -> None:
        """Render main dashboard content."""
        # Tab navigation
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview",
            "📈 Detailed Analysis",
            "🏭 State Comparison",
            "📍 Site Analysis",
            "📉 Data Quality",
            "🔍 Raw Data Explorer",
            "🔥 Advanced Analytics"
        ])

        with tab1:
            self._render_overview_tab()

        with tab2:
            self._render_analysis_tab()

        with tab3:
            self._render_comparison_tab()

        with tab4:
            self._render_site_analysis_tab()

        with tab5:
            self._render_quality_tab()

        with tab6:
            self._render_explorer_tab()

        with tab7:
            self._render_advanced_analytics_tab()

    def _render_overview_tab(self) -> None:
        """Render overview tab."""
        st.header("📊 Dashboard Overview")

        # KPI Metrics
        kpi_data = self.data_loader.get_kpi_metrics()

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric(
                label="Total Records",
                value=format_number(kpi_data.get('total_rows', 0)),
                delta="Real-time"
            )

        with col2:
            st.metric(
                label="States Covered",
                value=kpi_data.get('unique_states', 0)
            )

        with col3:
            avg_pm25 = kpi_data.get('avg_pm25', 0)
            st.metric(
                label="Avg PM2.5",
                value=f"{avg_pm25:.2f} µg/m³"
            )

        with col4:
            st.metric(
                label="Median PM2.5",
                value=f"{kpi_data.get('median_pm25', 0):.2f} µg/m³"
            )

        with col5:
            st.metric(
                label="Data Cleaned",
                value=format_number(
                    kpi_data.get('cleaned_records', 0)
                ),
                delta="Records"
            )

        with col6:
            quality_score = 95.5
            st.metric(
                label="Data Quality",
                value=f"{quality_score:.1f}%",
                delta="Excellent"
            )

        # Charts
        st.subheader("📈 Air Quality Trends")

        trends_data = self.data_loader.get_monthly_trends(st.session_state.filter_state)

        if not trends_data.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig_trend = self.visualizations.create_trend_chart(
                    trends_data,
                    st.session_state.filter_state.comparison_mode
                )
                st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

            with col2:
                # AQI Distribution
                where_clause = QueryBuilder.build_where_clause(
                    st.session_state.filter_state.selected_states,
                    st.session_state.filter_state.date_from,
                    st.session_state.filter_state.date_to
                )
                aqi_dist = self.data_loader.get_aqi_distribution(where_clause)

                if not aqi_dist.empty:
                    fig_aqi = self.visualizations.create_aqi_distribution(aqi_dist)
                    st.plotly_chart(fig_aqi, use_container_width=True, key="aqi_dist")

        # Current AQI Gauge
        if not trends_data.empty and 'avg_aqi' in trends_data.columns:
            st.subheader("🌈 Current Air Quality Index")
            current_aqi = trends_data['avg_aqi'].iloc[-1] if len(trends_data) > 0 else 0

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig_gauge = self.visualizations.create_gauge_chart(
                    current_aqi,
                    "Current AQI Level"
                )
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")

    def _render_analysis_tab(self) -> None:
        """Render detailed analysis tab."""
        st.header("📈 Detailed Analysis")

        trends_data = self.data_loader.get_monthly_trends(st.session_state.filter_state)

        if trends_data.empty:
            st.warning("No data available for selected filters")
            return

        # Anomaly detection
        if st.session_state.filter_state.show_anomalies and 'avg_pm25' in trends_data.columns:
            trends_data = self.anomaly_detector.detect_anomalies(
                trends_data,
                'avg_pm25',
                method='zscore',
                threshold=2.5
            )

            anomalies = trends_data[
                trends_data['is_anomaly']] if 'is_anomaly' in trends_data.columns else pd.DataFrame()
            if not anomalies.empty:
                st.warning(f"🔍 Detected {len(anomalies)} anomalies in the data")

                with st.expander("View Anomalies"):
                    st.dataframe(
                        anomalies[['month', 'avg_pm25', 'anomaly_score']],
                        use_container_width=True
                    )

        # Statistical Summary
        st.subheader("📊 Statistical Summary")

        numeric_cols = trends_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_stats = trends_data[numeric_cols].describe()
            st.dataframe(summary_stats, use_container_width=True)

        # Time Series Decomposition
        if 'avg_pm25' in trends_data.columns and len(trends_data) > 3:
            st.subheader("📈 PM2.5 Trend Analysis")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trends_data['month'],
                y=trends_data['avg_pm25'],
                mode='lines+markers',
                name='PM2.5',
                line=dict(width=2)
            ))

            # Add moving average
            if len(trends_data) > 3:
                trends_data['ma3'] = trends_data['avg_pm25'].rolling(window=3, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=trends_data['month'],
                    y=trends_data['ma3'],
                    mode='lines',
                    name='3-Month Moving Avg',
                    line=dict(dash='dash')
                ))

            fig.update_layout(
                title='PM2.5 Levels with Moving Average',
                xaxis_title='Month',
                yaxis_title='PM2.5 (µg/m³)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True, key="ma_chart")

    def _render_comparison_tab(self) -> None:
        """Render state comparison tab."""
        st.header("🏭 State-by-State Comparison")

        comparison_data = self.data_loader.get_state_comparison(st.session_state.filter_state)

        if comparison_data.empty:
            st.warning("No data available for comparison")
            return

        # Heatmap
        st.subheader("🗺️ State Air Quality Heatmap")
        fig_heatmap = self.visualizations.create_heatmap(comparison_data.head(10))
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")

        # Best and Worst States
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Best Air Quality States")
            best_states = comparison_data.nsmallest(5, 'avg_pm25')[['State Name', 'avg_pm25', 'pct_good_days']]
            st.dataframe(best_states, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("⚠️ Worst Air Quality States")
            worst_states = comparison_data.nlargest(5, 'avg_pm25')[['State Name', 'avg_pm25', 'unhealthy_days']]
            st.dataframe(worst_states, use_container_width=True, hide_index=True)

    def _render_site_analysis_tab(self) -> None:
        """Render site analysis tab."""
        st.header("📍 Monitoring Site Analysis")

        site_data = self.data_loader.get_site_analysis(st.session_state.filter_state)

        if site_data.empty:
            st.warning("No site data available")
            return

        # Map visualization
        st.subheader("🗺️ Site Locations")
        fig_map = self.visualizations.create_site_map(site_data)
        st.plotly_chart(fig_map, use_container_width=True, key="site_map")

        # Site statistics
        st.subheader("📊 Top Monitoring Sites by PM2.5")
        display_cols = ['site_name', 'state', 'county', 'avg_pm25', 'avg_aqi', 'measurements']
        st.dataframe(site_data[display_cols].head(10), use_container_width=True, hide_index=True)

    def _render_quality_tab(self) -> None:
        """Render data quality tab."""
        st.header("📉 Data Quality Report")

        quality_metrics = self.data_loader.get_data_quality_metrics()

        # Quality Score Gauge
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig_gauge = self.visualizations.create_gauge_chart(
                quality_metrics.quality_score,
                "Overall Data Quality Score",
                max_value=100
            )
            st.plotly_chart(fig_gauge, use_container_width=True, key="quality_gauge")

        # Metrics
        st.subheader("📊 Quality Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", format_number(quality_metrics.total_records))

        with col2:
            null_pct = (quality_metrics.null_count / quality_metrics.total_records * 100) if quality_metrics.total_records > 0 else 0
            st.metric("NULL Values", format_number(quality_metrics.null_count), f"{null_pct:.2f}%")

        with col3:
            neg_pct = (quality_metrics.negative_count / quality_metrics.total_records * 100) if quality_metrics.total_records > 0 else 0
            st.metric("Negative Values Fixed", format_number(quality_metrics.negative_count), f"{neg_pct:.2f}%")

        with col4:
            outlier_pct = (quality_metrics.outlier_count / quality_metrics.total_records * 100) if quality_metrics.total_records > 0 else 0
            st.metric("Outliers", format_number(quality_metrics.outlier_count), f"{outlier_pct:.2f}%")

    def _render_explorer_tab(self) -> None:
        """Render raw data explorer tab."""
        st.header("🔍 Raw Data Explorer")

        col1, col2 = st.columns(2)

        with col1:
            rows_limit = st.selectbox("Rows to display", [100, 500, 1000], key="rows_limit")

        with col2:
            search_term = st.text_input("Search", placeholder="Filter data...", key="search")

        # Get data
        raw_data = self.data_loader.get_raw_data(st.session_state.filter_state, limit=rows_limit)

        if not raw_data.empty:
            # Apply search
            if search_term:
                mask = raw_data.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                raw_data = raw_data[mask]

            st.subheader(f"📊 Showing {len(raw_data)} records")
            st.dataframe(raw_data, use_container_width=True, height=400)

            # Download button
            csv = raw_data.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"epa_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data available")

    def _render_advanced_analytics_tab(self) -> None:
        """Render advanced analytics tab with Window Functions (Tag14 Homework)."""
        st.header("🔥 Advanced Analytics - Window Functions")
        st.markdown("**LAG | ROW_NUMBER | Rolling Windows | Subqueries**")

        # Sub-tabs for different analyses
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
            "🚨 Alarm-Tage",
            "📊 Quartals-Trend",
            "🏆 Worst Days",
            "📈 Rolling Average",
            "🎯 Bonus Challenges"
        ])

        # =====================================================================
        # TAB 1: ALARM-TAGE (LAG Window Function)
        # =====================================================================
        with analysis_tab1:
            st.subheader("🚨 Alarm-Tage: PM2.5 Anstiege >100%")
            st.markdown("""
            **Business Frage:** An welchen Tagen ist PM2.5 um mehr als 100% gestiegen?
            
            Diese Analyse nutzt die `LAG()` Window Function um den Vortageswert zu holen
            und prozentuale Veränderungen zu berechnen.
            """)

            alarm_df = self.data_loader.get_alarm_days(st.session_state.filter_state)

            if not alarm_df.empty:
                # KPI Row
                col1, col2, col3, col4 = st.columns(4)

                kritisch_count = len(alarm_df[alarm_df['alarm_level'] == 'KRITISCH'])
                warnung_count = len(alarm_df[alarm_df['alarm_level'] == 'WARNUNG'])
                max_sprung = alarm_df['prozent_change'].max() if 'prozent_change' in alarm_df.columns else 0
                affected_states = alarm_df['State Name'].nunique()

                with col1:
                    st.metric(
                        label="🔴 KRITISCH (>200%)",
                        value=kritisch_count,
                        delta="Tage" if kritisch_count > 0 else None,
                        delta_color="inverse"
                    )

                with col2:
                    st.metric(
                        label="🟡 WARNUNG (>100%)",
                        value=warnung_count,
                        delta="Tage",
                        delta_color="inverse"
                    )

                with col3:
                    st.metric(
                        label="📈 Max Anstieg",
                        value=f"{max_sprung:.1f}%",
                        delta="Größter Sprung"
                    )

                with col4:
                    st.metric(
                        label="🗺️ Betroffene Staaten",
                        value=affected_states
                    )

                # Filter by alarm level
                st.markdown("---")
                alarm_filter = st.multiselect(
                    "Filter nach Alarm-Level:",
                    options=['KRITISCH', 'WARNUNG'],
                    default=['KRITISCH', 'WARNUNG'],
                    key="alarm_filter"
                )

                filtered_alarm = alarm_df[alarm_df['alarm_level'].isin(alarm_filter)]

                # Display Table
                st.dataframe(
                    filtered_alarm,
                    column_config={
                        "State Name": st.column_config.TextColumn("🗺️ Staat"),
                        "Date Local": st.column_config.DateColumn("📅 Datum"),
                        "heute": st.column_config.NumberColumn("Heute (µg/m³)", format="%.2f"),
                        "gestern": st.column_config.NumberColumn("Gestern (µg/m³)", format="%.2f"),
                        "prozent_change": st.column_config.NumberColumn(
                            "📈 Anstieg %",
                            format="%.1f%%",
                            help="Prozentuale Veränderung zum Vortag"
                        ),
                        "alarm_level": st.column_config.TextColumn("⚠️ Level")
                    },
                    use_container_width=True,
                    hide_index=True
                )

                # Visualization
                if len(filtered_alarm) > 0:
                    fig = px.bar(
                        filtered_alarm.head(20),
                        x='State Name',
                        y='prozent_change',
                        color='alarm_level',
                        color_discrete_map={'KRITISCH': '#ff4444', 'WARNUNG': '#ffaa00'},
                        title='Top 20 Alarm-Tage nach Anstieg %',
                        labels={'prozent_change': 'Anstieg (%)', 'State Name': 'Staat'}
                    )
                    st.plotly_chart(fig, use_container_width=True, key="alarm_chart")
            else:
                st.info("Keine Alarm-Tage mit >100% Anstieg gefunden.")

            # Show SQL Query
            with st.expander("📝 SQL Query anzeigen"):
                st.code("""
WITH daily_avg AS (
    SELECT "State Name", "Date Local",
           AVG(pm25_cleaned) as pm25
    FROM data GROUP BY "State Name", "Date Local"
),
with_yesterday AS (
    SELECT *, 
           LAG(pm25) OVER (
               PARTITION BY "State Name" ORDER BY "Date Local"
           ) as gestern
    FROM daily_avg
)
SELECT *, 
       (heute - gestern) / gestern * 100 as prozent_change,
       CASE WHEN prozent > 200 THEN 'KRITISCH'
            WHEN prozent > 100 THEN 'WARNUNG'
       END as alarm_level
FROM with_yesterday
WHERE prozent_change > 100
                """, language="sql")

        # =====================================================================
        # TAB 2: QUARTALS-TREND (Q1 vs Q4)
        # =====================================================================
        with analysis_tab2:
            st.subheader("📊 Quartals-Trend: Q1 vs Q4 Vergleich")
            st.markdown("""
            **Business Frage:** Welche Staaten haben sich über das Jahr verbessert oder verschlechtert?
            
            Vergleicht den Durchschnitt von Q1 (Jan-März) mit Q4 (Okt-Dez).
            """)

            trend_df = self.data_loader.get_state_quarterly_trend()

            if not trend_df.empty:
                # Summary KPIs
                col1, col2, col3 = st.columns(3)

                verbessert = len(trend_df[trend_df['kategorie'] == 'Verbessert'])
                verschlechtert = len(trend_df[trend_df['kategorie'] == 'Verschlechtert'])
                stabil = len(trend_df[trend_df['kategorie'] == 'Stabil'])

                with col1:
                    st.metric("✅ Verbessert", verbessert, delta="Staaten", delta_color="normal")

                with col2:
                    st.metric("❌ Verschlechtert", verschlechtert, delta="Staaten", delta_color="inverse")

                with col3:
                    st.metric("➖ Stabil", stabil, delta="Staaten")

                st.markdown("---")

                # Dual view
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Bar chart Q1 vs Q4
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Q1 (Jan-März)',
                        x=trend_df['State Name'].head(15),
                        y=trend_df['q1_avg'].head(15),
                        marker_color='#3498db'
                    ))
                    fig.add_trace(go.Bar(
                        name='Q4 (Okt-Dez)',
                        x=trend_df['State Name'].head(15),
                        y=trend_df['q4_avg'].head(15),
                        marker_color='#e74c3c'
                    ))
                    fig.update_layout(
                        title='Q1 vs Q4 PM2.5 Durchschnitt (Top 15 Staaten)',
                        barmode='group',
                        xaxis_title='Staat',
                        yaxis_title='PM2.5 (µg/m³)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True, key="q1q4_chart")

                with col2:
                    st.subheader("🏆 Top Verbesserungen")
                    improved = trend_df[trend_df['kategorie'] == 'Verbessert'].head(5)
                    for _, row in improved.iterrows():
                        st.write(f"✅ **{row['State Name']}**: {row['prozent_change']:.1f}%")

                    st.subheader("📉 Top Verschlechterungen")
                    worsened = trend_df[trend_df['kategorie'] == 'Verschlechtert'].tail(5)
                    for _, row in worsened.iterrows():
                        st.write(f"❌ **{row['State Name']}**: +{abs(row['prozent_change']):.1f}%")

                # Full table
                with st.expander("📋 Alle Staaten anzeigen"):
                    st.dataframe(
                        trend_df,
                        column_config={
                            "State Name": "Staat",
                            "q1_avg": st.column_config.NumberColumn("Q1 Avg", format="%.2f"),
                            "q4_avg": st.column_config.NumberColumn("Q4 Avg", format="%.2f"),
                            "diff": st.column_config.NumberColumn("Differenz", format="%.2f"),
                            "prozent_change": st.column_config.NumberColumn("% Änderung", format="%.1f%%"),
                            "improvement_rank": "Rang",
                            "kategorie": "Status"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.warning("Keine Quartals-Daten verfügbar.")

        # =====================================================================
        # TAB 3: WORST DAYS PER STATE (ROW_NUMBER)
        # =====================================================================
        with analysis_tab3:
            st.subheader("🏆 Top 3 Schlechteste Tage pro Staat")
            st.markdown("""
            **Business Frage:** Welche waren die 3 schlimmsten Luftqualitäts-Tage in jedem Staat?
            
            Nutzt `ROW_NUMBER()` für das Ranking und `LAG()` für Vortagesvergleich.
            """)

            worst_df = self.data_loader.get_worst_days_per_state(st.session_state.filter_state)

            if not worst_df.empty:
                # Summary
                col1, col2, col3 = st.columns(3)

                with col1:
                    max_pm25 = worst_df['pm25'].max()
                    max_state = worst_df.loc[worst_df['pm25'].idxmax(), 'State Name']
                    st.metric("📈 Höchster PM2.5 Wert", f"{max_pm25:.2f} µg/m³", delta=max_state)

                with col2:
                    sudden_jumps = len(worst_df[worst_df['sprung_typ'] == 'Plötzlich'])
                    st.metric("⚡ Plötzliche Sprünge", sudden_jumps, delta=">20 µg/m³")

                with col3:
                    avg_over = worst_df['ueber_durchschnitt'].mean()
                    st.metric("📊 Ø über US-Schnitt", f"{avg_over:.2f} µg/m³")

                # Table
                st.dataframe(
                    worst_df,
                    column_config={
                        "State Name": "🗺️ Staat",
                        "Date Local": st.column_config.DateColumn("📅 Datum"),
                        "pm25": st.column_config.NumberColumn("PM2.5", format="%.2f"),
                        "rang": "Rang",
                        "vortag": st.column_config.NumberColumn("Vortag", format="%.2f"),
                        "sprung": st.column_config.NumberColumn("Sprung", format="%.2f"),
                        "ueber_durchschnitt": st.column_config.NumberColumn("Über Ø", format="%.2f"),
                        "sprung_typ": "⚡ Typ"
                    },
                    use_container_width=True,
                    hide_index=True
                )

                # Visualization
                fig = px.scatter(
                    worst_df,
                    x='State Name',
                    y='pm25',
                    size='pm25',
                    color='sprung_typ',
                    color_discrete_map={'Plötzlich': '#e74c3c', 'Normal': '#3498db'},
                    title='Top 3 Schlechteste Tage pro Staat',
                    labels={'pm25': 'PM2.5 (µg/m³)', 'State Name': 'Staat'}
                )
                st.plotly_chart(fig, use_container_width=True, key="worst_scatter")
            else:
                st.warning("Keine Daten verfügbar.")

        # =====================================================================
        # TAB 4: ROLLING AVERAGE (Window Frame)
        # =====================================================================
        with analysis_tab4:
            st.subheader("📈 7-Tage Rolling Average")
            st.markdown("""
            **Business Frage:** Wie sieht der geglättete Trend aus (ohne tägliches Rauschen)?
            
            Nutzt `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` für den gleitenden Durchschnitt.
            """)

            # State selector for rolling average
            states_list = self.data_loader.get_states_list()
            if states_list:
                selected_state_rolling = st.selectbox(
                    "🗺️ Staat für Rolling Average wählen:",
                    options=states_list,
                    index=0,
                    key="rolling_state"
                )

                rolling_df = self.data_loader.get_rolling_average(selected_state_rolling)

                if not rolling_df.empty:
                    # Create dual-line chart
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=rolling_df['Date Local'],
                        y=rolling_df['daily_pm25'],
                        mode='lines',
                        name='Täglicher Wert',
                        line=dict(color='rgba(52, 152, 219, 0.5)', width=1),
                        fill='tozeroy',
                        fillcolor='rgba(52, 152, 219, 0.1)'
                    ))

                    fig.add_trace(go.Scatter(
                        x=rolling_df['Date Local'],
                        y=rolling_df['rolling_7day'],
                        mode='lines',
                        name='7-Tage Rolling Avg',
                        line=dict(color='#e74c3c', width=3)
                    ))

                    fig.update_layout(
                        title=f'PM2.5 Trend für {selected_state_rolling}',
                        xaxis_title='Datum',
                        yaxis_title='PM2.5 (µg/m³)',
                        height=450,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True, key="rolling_chart")

                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("📊 Ø Täglicher Wert", f"{rolling_df['daily_pm25'].mean():.2f}")

                    with col2:
                        st.metric("📈 Max Rolling Avg", f"{rolling_df['rolling_7day'].max():.2f}")

                    with col3:
                        above_trend = len(rolling_df[rolling_df['trend_position'] == 'Über Trend'])
                        st.metric("⬆️ Tage über Trend", above_trend)

                    with col4:
                        below_trend = len(rolling_df[rolling_df['trend_position'] == 'Unter Trend'])
                        st.metric("⬇️ Tage unter Trend", below_trend)
                else:
                    st.warning(f"Keine Daten für {selected_state_rolling} verfügbar.")
            else:
                st.warning("Keine Staaten gefunden.")

        # =====================================================================
        # TAB 5: BONUS CHALLENGES
        # =====================================================================
        with analysis_tab5:
            st.subheader("🎯 Bonus Challenges")

            bonus_tab1, bonus_tab2, bonus_tab3 = st.tabs([
                "📅 Wochentag-Analyse",
                "🏃 Good Air Streaks",
                "📊 Anomaly Scores"
            ])

            # Bonus 1: Weekday Analysis
            with bonus_tab1:
                st.markdown("**Challenge 1:** An welchen Wochentagen ist die Luft am schlechtesten?")

                weekday_df = self.data_loader.get_weekday_analysis()

                if not weekday_df.empty:
                    fig = px.bar(
                        weekday_df,
                        x='wochentag',
                        y='avg_pm25',
                        color='avg_pm25',
                        color_continuous_scale='RdYlGn_r',
                        title='Durchschnittlicher PM2.5 nach Wochentag',
                        labels={'avg_pm25': 'PM2.5 (µg/m³)', 'wochentag': 'Wochentag'}
                    )
                    st.plotly_chart(fig, use_container_width=True, key="weekday_chart")

                    # Table
                    st.dataframe(weekday_df, use_container_width=True, hide_index=True)

            # Bonus 2: Good Air Streaks
            with bonus_tab2:
                st.markdown("**Challenge 2:** Längste Serie von 'Good Air Quality' Tagen (PM2.5 < 12)")

                streak_df = self.data_loader.get_good_air_streaks(top_n=15)

                if not streak_df.empty:
                    fig = px.bar(
                        streak_df,
                        x='State Name',
                        y='Tage',
                        color='Tage',
                        color_continuous_scale='Greens',
                        title='Längste Serien guter Luftqualität pro Staat',
                        labels={'Tage': 'Anzahl Tage'}
                    )
                    st.plotly_chart(fig, use_container_width=True, key="streak_chart")

                    st.dataframe(streak_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Keine Streak-Daten verfügbar.")

            # Bonus 3: Anomaly Scores
            with bonus_tab3:
                st.markdown("**Challenge 3:** Z-Score Anomalie-Erkennung")

                anomaly_df = self.data_loader.get_anomaly_scores(st.session_state.filter_state)

                if not anomaly_df.empty:
                    # Filter
                    anomaly_filter = st.multiselect(
                        "Anomalie-Level filtern:",
                        options=['Extreme Anomalie', 'Starke Anomalie', 'Leichte Anomalie', 'Normal'],
                        default=['Extreme Anomalie', 'Starke Anomalie'],
                        key="anomaly_level_filter"
                    )

                    filtered_anomaly = anomaly_df[anomaly_df['anomaly_level'].isin(anomaly_filter)]

                    fig = px.scatter(
                        filtered_anomaly.head(50),
                        x='Date Local',
                        y='z_score',
                        color='anomaly_level',
                        size=abs(filtered_anomaly.head(50)['z_score']),
                        hover_data=['State Name', 'pm25', 'state_avg'],
                        title='Z-Score Anomalien',
                        labels={'z_score': 'Z-Score', 'Date Local': 'Datum'}
                    )
                    fig.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="2σ")
                    fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="3σ")
                    fig.add_hline(y=-2, line_dash="dash", line_color="orange")
                    fig.add_hline(y=-3, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True, key="anomaly_scatter")

                    st.dataframe(
                        filtered_anomaly,
                        column_config={
                            "State Name": "Staat",
                            "Date Local": st.column_config.DateColumn("Datum"),
                            "pm25": st.column_config.NumberColumn("PM2.5", format="%.2f"),
                            "state_avg": st.column_config.NumberColumn("Staats-Ø", format="%.2f"),
                            "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                            "anomaly_level": "Level"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Keine Anomalie-Daten verfügbar.")

    def _render_footer(self) -> None:
        """Render dashboard footer."""
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; color: gray;'>
                <p>📊 EPA Air Quality Dashboard v2.1 | Built with Streamlit, DuckDB & Plotly</p>
                <p>🔥 Includes Advanced Analytics with Window Functions (Tag14 Homework)</p>
                <p>© 2024 Sebastian Kühnrich | Professional Edition</p>
                <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configuration
    import os
    from pathlib import Path

    BASE_DIR = Path(__file__).parent

    # Try parquet first, fallback to CSV
    DATA_PATH_PARQUET = BASE_DIR / 'Data' / 'daily_88101_2024_cleaned.parquet'
    DATA_PATH_CSV = BASE_DIR / 'Data' / 'daily_88101_2024_cleaned.csv'

    if os.path.exists(str(DATA_PATH_PARQUET)):
        DATA_PATH = str(DATA_PATH_PARQUET)
        print(f"✅ Using Parquet file: {DATA_PATH}")
    elif os.path.exists(str(DATA_PATH_CSV)):
        DATA_PATH = str(DATA_PATH_CSV)
        print(f"✅ Using CSV file: {DATA_PATH}")
    else:
        st.error(f"❌ Datendatei nicht gefunden!")
        st.info("Gesuchte Dateien:")
        st.code(f"- {DATA_PATH_PARQUET}\n- {DATA_PATH_CSV}")
        st.stop()

    # Run dashboard
    dashboard = EPADashboard(DATA_PATH)
    dashboard.run()