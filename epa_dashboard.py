# =============================================================================
# EPA AIR QUALITY DASHBOARD - Tag 14 Projekt
# =============================================================================

import streamlit as st
import duckdb
import pandas as pd
from datetime import datetime

# -----------------------------------------------------------------------------
# KONFIGURATION
# -----------------------------------------------------------------------------

# Dein Dateipfad (ANPASSEN!)
DATA_PATH = r'C:\Users\sebas\PycharmProjects\Big_Data_Umwelbehörde\Data\daily_88101_2024.csv'


# DuckDB Connection mit Caching
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
# TEIL 1: KPI METRICS
# -----------------------------------------------------------------------------

st.header("📈 Key Performance Indicators")


@st.cache_data
def get_kpi_data():
    return con.execute(f"""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT "State Name") as unique_states,
            SUM(CASE WHEN "Arithmetic Mean" < 0 THEN 1 ELSE 0 END) as negative_fixed,
            ROUND(AVG(GREATEST(COALESCE("Arithmetic Mean", 0), 0)), 2) as avg_pm25,
            SUM(CASE WHEN "AQI" IS NULL THEN 1 ELSE 0 END) as null_aqi_fixed,
            SUM(CASE WHEN "AQI" > 500 THEN 1 ELSE 0 END) as extreme_aqi_capped
        FROM '{DATA_PATH}'
    """).fetchdf()


kpi_df = get_kpi_data()

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

# Zweite Reihe KPIs
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        label="🔢 NULL AQI Calculated",
        value=f"{kpi_df['null_aqi_fixed'].iloc[0]:,.0f}",
        delta="EPA Formula",
        delta_color="normal"
    )

with col6:
    st.metric(
        label="⚠️ Extreme AQI Capped",
        value=f"{kpi_df['extreme_aqi_capped'].iloc[0]:,.0f}",
        delta="Max 500",
        delta_color="normal"
    )

# -----------------------------------------------------------------------------
# TEIL 2: SIDEBAR FILTERS
# -----------------------------------------------------------------------------

st.sidebar.header("🔎 Filter")


# Hole alle States für Dropdown
@st.cache_data
def get_states():
    return con.execute(f"""
        SELECT DISTINCT "State Name"
        FROM '{DATA_PATH}'
        WHERE "State Name" IS NOT NULL
        ORDER BY "State Name"
    """).fetchdf()


states_df = get_states()
state_list = ['All States'] + states_df['State Name'].tolist()

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


# Baue die WHERE Clause dynamisch
def build_where_clause(state, date_from, date_to):
    conditions = []

    if state != "All States":
        conditions.append(f'"State Name" = \'{state}\'')

    conditions.append(f'"Date Local" >= \'{date_from}\'')
    conditions.append(f'"Date Local" <= \'{date_to}\'')

    if conditions:
        return "WHERE " + " AND ".join(conditions)
    return ""


where_clause = build_where_clause(selected_state, date_from, date_to)

# -----------------------------------------------------------------------------
# TEIL 3: TREND CHART
# -----------------------------------------------------------------------------

st.header("📈 Air Quality Trend 2024")


@st.cache_data
def get_monthly_trend(where_clause):
    return con.execute(f"""
        WITH cleaned_data AS (
            SELECT
                STRFTIME("Date Local"::DATE, '%Y-%m') as month,
                GREATEST(COALESCE("Arithmetic Mean", 0), 0) as pm25_cleaned,
                LEAST(
                    GREATEST(
                        COALESCE(
                            "AQI",
                            CASE
                                WHEN "Arithmetic Mean" <= 12.0 THEN "Arithmetic Mean" * 50.0 / 12.0
                                WHEN "Arithmetic Mean" <= 35.4 THEN 50 + ("Arithmetic Mean" - 12.0) * 49.0 / 23.4
                                WHEN "Arithmetic Mean" <= 55.4 THEN 100 + ("Arithmetic Mean" - 35.4) * 49.0 / 20.0
                                WHEN "Arithmetic Mean" <= 150.4 THEN 150 + ("Arithmetic Mean" - 55.4) * 49.0 / 95.0
                                WHEN "Arithmetic Mean" <= 250.4 THEN 200 + ("Arithmetic Mean" - 150.4) * 99.0 / 100.0
                                ELSE 300 + ("Arithmetic Mean" - 250.4) * 199.0 / 250.0
                            END
                        ), 0
                    ), 500
                ) as aqi_cleaned
            FROM '{DATA_PATH}'
            {where_clause}
        )
        SELECT
            month,
            ROUND(AVG(pm25_cleaned), 2) as avg_pm25,
            ROUND(AVG(aqi_cleaned), 1) as avg_aqi,
            SUM(CASE WHEN aqi_cleaned > 100 THEN 1 ELSE 0 END) as unhealthy_days
        FROM cleaned_data
        GROUP BY month
        ORDER BY month
    """).fetchdf()


trend_df = get_monthly_trend(where_clause)

if not trend_df.empty:
    # Zwei Spalten für Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💨 PM2.5 Trend")
        chart_data = trend_df.set_index('month')
        st.line_chart(chart_data['avg_pm25'], use_container_width=True)

    with col2:
        st.subheader("🌈 AQI Trend")
        st.line_chart(chart_data['avg_aqi'], use_container_width=True)

    # Zeige auch die Tabelle darunter
    with st.expander("📋 Show Data Table"):
        st.dataframe(trend_df, use_container_width=True)
else:
    st.warning("No data for selected filters")

# -----------------------------------------------------------------------------
# TEIL 4: CLEANUP SUMMARY CHART
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
                    WHEN "Arithmetic Mean" IS NULL THEN 'PM2.5 NULL replaced'
                    WHEN "Arithmetic Mean" < 0 THEN 'Negative corrected'
                    WHEN "AQI" > 500 THEN 'AQI capped at 500'
                    WHEN "AQI" IS NULL THEN 'AQI calculated'
                    ELSE 'Unchanged'
                END as cleanup_action
            FROM '{DATA_PATH}'
            {where_clause}
        )
        GROUP BY cleanup_action
        ORDER BY count DESC
    """).fetchdf()


cleanup_df = get_cleanup_summary(where_clause)

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
        total = cleanup_df['count'].sum()
        for _, row in cleanup_df.iterrows():
            percentage = row['count'] / total * 100
            st.write(f"**{row['cleanup_action']}:**")
            st.write(f"  {row['count']:,} ({percentage:.2f}%)")

# -----------------------------------------------------------------------------
# TEIL 5: TOP POLLUTED STATES TABLE
# -----------------------------------------------------------------------------

st.header("🏭 States with Worst Air Quality")


@st.cache_data
def get_worst_states():
    return con.execute(f"""
        WITH cleaned_data AS (
            SELECT
                "State Name",
                GREATEST(COALESCE("Arithmetic Mean", 0), 0) as pm25_cleaned,
                LEAST(
                    GREATEST(
                        COALESCE(
                            "AQI",
                            CASE
                                WHEN "Arithmetic Mean" <= 12.0 THEN "Arithmetic Mean" * 50.0 / 12.0
                                WHEN "Arithmetic Mean" <= 35.4 THEN 50 + ("Arithmetic Mean" - 12.0) * 49.0 / 23.4
                                ELSE 100
                            END
                        ), 0
                    ), 500
                ) as aqi_cleaned
            FROM '{DATA_PATH}'
        )
        SELECT
            "State Name",
            COUNT(*) as measurements,
            ROUND(AVG(pm25_cleaned), 2) as avg_pm25,
            ROUND(MAX(pm25_cleaned), 2) as max_pm25,
            SUM(CASE WHEN aqi_cleaned > 100 THEN 1 ELSE 0 END) as unhealthy_days
        FROM cleaned_data
        GROUP BY "State Name"
        HAVING COUNT(*) > 100
        ORDER BY avg_pm25 DESC
        LIMIT 10
    """).fetchdf()


worst_states_df = get_worst_states()

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

# -----------------------------------------------------------------------------
# BONUS: AQI DISTRIBUTION PIE CHART
# -----------------------------------------------------------------------------

st.header("🌈 AQI Category Distribution")


@st.cache_data
def get_aqi_distribution(where_clause):
    return con.execute(f"""
        WITH cleaned_data AS (
            SELECT
                LEAST(
                    GREATEST(
                        COALESCE(
                            "AQI",
                            CASE
                                WHEN "Arithmetic Mean" <= 12.0 THEN "Arithmetic Mean" * 50.0 / 12.0
                                WHEN "Arithmetic Mean" <= 35.4 THEN 50 + ("Arithmetic Mean" - 12.0) * 49.0 / 23.4
                                ELSE 100
                            END
                        ), 0
                    ), 500
                ) as aqi_cleaned
            FROM '{DATA_PATH}'
            {where_clause}
        )
        SELECT
            CASE
                WHEN aqi_cleaned <= 50 THEN '1. Good (0-50)'
                WHEN aqi_cleaned <= 100 THEN '2. Moderate (51-100)'
                WHEN aqi_cleaned <= 150 THEN '3. Unhealthy Sensitive (101-150)'
                WHEN aqi_cleaned <= 200 THEN '4. Unhealthy (151-200)'
                WHEN aqi_cleaned <= 300 THEN '5. Very Unhealthy (201-300)'
                ELSE '6. Hazardous (301-500)'
            END as category,
            COUNT(*) as count
        FROM cleaned_data
        GROUP BY category
        ORDER BY category
    """).fetchdf()


aqi_dist_df = get_aqi_distribution(where_clause)

if not aqi_dist_df.empty:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.bar_chart(aqi_dist_df.set_index('category'), use_container_width=True)

    with col2:
        st.subheader("📊 Percentages")
        total = aqi_dist_df['count'].sum()
        for _, row in aqi_dist_df.iterrows():
            percentage = row['count'] / total * 100
            st.write(f"**{row['category']}:**")
            st.write(f"  {percentage:.1f}%")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 EPA Air Quality Dashboard | Built with Streamlit & DuckDB</p>
        <p>Data Source: <a href='https://aqs.epa.gov/aqsweb/airdata/download_files.html'>EPA AirData</a></p>
        <p>Created by: Big Data Student | Tag 14 Projekt</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.info("""
    **About this Dashboard**

    This dashboard visualizes 738,478 PM2.5 air quality 
    measurements from the EPA for 2024.

    **Data Cleanup Applied:**
    - NULL values → Calculated
    - Negative values → Set to 0
    - Extreme AQI (>500) → Capped

    **Filter Options:**
    - Select specific state
    - Filter by date range
    - View monthly trends
""")

# Performance Info
st.sidebar.markdown("---")
st.sidebar.success(f"""
    **📈 Current Selection:**
    - State: {selected_state}
    - Date Range: {date_from} to {date_to}

    Refresh page to reload data!
""")