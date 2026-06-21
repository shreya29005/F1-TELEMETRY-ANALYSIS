import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import html
import logging
import re
import unicodedata
from data_fetch import fetch_driver_telemetry, fetch_driver_top5_telemetry, enable_fastf1_cache
from feature_engg import build_feature_dataset, extract_turn_features

logger = logging.getLogger(__name__)

# Default analysis configuration
DEFAULT_YEAR = 2025
DEFAULT_GP = "Bahrain"
DEFAULT_SESSION = "Q"
DEFAULT_DRIVERS = [
    "VER", "LEC", "HAM", "RUS", "SAI", "NOR", "ALO", "PER", "PIA", "GAS",
    "STR", "TSU", "ALB", "OCO", "HUL", "MAG", "BOT", "ZHO",
]
GP_OPTIONS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami", "Monaco",
    "Canada", "Silverstone", "Hungary", "Belgium", "Netherlands", "Italy",
    "Singapore", "United States", "Mexico City", "Brazil", "Las Vegas", "Abu Dhabi",
]
SESSION_LABELS = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
SESSION_VALUES = ["FP1", "FP2", "FP3", "Q", "R"]
DEFAULT_START_DISTANCE = 300
DEFAULT_END_DISTANCE = 600
CACHE_DIR = "cache"
CLUSTER_FEATURE_COLUMNS = [
    "Entry_Speed",
    "Apex_Speed",
    "Exit_Speed",
    "Speed_Loss",
    "Speed_Recovery",
    "Avg_Speed_MicroSector",
    "Speed_Std_MicroSector",
    "Braking_Pct",
    "Throttle_Pct",
    "Full_Throttle_Pct",
    "Braking_Zone_Length",
    "Trail_Braking_Index",
    "Throttle_Commitment_Index",
    "Corner_Aggression_Score",
    "Smoothness_Index",
]
DRIVER_PROFILE_NUMERIC_COLUMNS = [
    "Sample_Count",
    "Entry_Speed",
    "Apex_Speed",
    "Exit_Speed",
    "Speed_Loss",
    "Speed_Recovery",
    "Braking_Pct",
    "Throttle_Pct",
    "Full_Throttle_Pct",
    "Braking_Zone_Length",
    "Trail_Braking_Index",
    "Throttle_Commitment_Index",
    "Corner_Aggression_Score",
    "Smoothness_Index",
]
CONSISTENCY_METRICS = [
    "Apex_Speed",
    "Braking_Pct",
    "Brake_Start_Distance",
    "Speed_Loss",
    "Throttle_Commitment_Index",
]

def render_landing_page():
    try:
        with open("style.css", "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
    st.markdown(
        """
        <div style="
            min-height:90vh;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            padding:60px 40px;
            text-align:center;
        ">
            <h1 style="color:#ffffff;font-size:3rem;font-weight:800;margin-bottom:12px;">
                F1 Driving Style Clustering
            </h1>
            <p style="color:#C8102E;font-size:1.2rem;font-weight:600;margin-bottom:16px;">
                Micro-sector telemetry analysis using FastF1 and unsupervised ML
            </p>
            <p style="color:#aaaaaa;font-size:1rem;max-width:640px;margin-bottom:48px;">
                Compare how drivers brake, carry speed, and return to throttle across any distance window on circuit.
            </p>
            <div style="display:flex;gap:24px;justify-content:center;flex-wrap:wrap;max-width:960px;margin-bottom:48px;">
                <div style="background:#1a1a1a;border-top:3px solid #C8102E;padding:24px;border-radius:4px;flex:1;min-width:260px;max-width:300px;">
                    <div style="color:#ffffff;font-weight:700;font-size:1rem;margin-bottom:8px;">Telemetry Analysis</div>
                    <div style="color:#aaaaaa;font-size:0.9rem;">Fastest lap and top-5 lap consistency across micro-sectors</div>
                </div>
                <div style="background:#1a1a1a;border-top:3px solid #C8102E;padding:24px;border-radius:4px;flex:1;min-width:260px;max-width:300px;">
                    <div style="color:#ffffff;font-weight:700;font-size:1rem;margin-bottom:8px;">K-Means Clustering</div>
                    <div style="color:#aaaaaa;font-size:0.9rem;">Unsupervised driving style classification with PCA biplot</div>
                </div>
                <div style="background:#1a1a1a;border-top:3px solid #C8102E;padding:24px;border-radius:4px;flex:1;min-width:260px;max-width:300px;">
                    <div style="color:#ffffff;font-weight:700;font-size:1rem;margin-bottom:8px;">Consistency Scoring</div>
                    <div style="color:#aaaaaa;font-size:0.9rem;">Lap-over-lap repeatability metrics per driver and sector</div>
                </div>
            </div>
            <div class="track-container">
                <svg class="speed-lines" width="240" height="24" viewBox="0 0 120 12">
                    <line x1="0" y1="3" x2="120" y2="3" stroke="#C8102E" stroke-width="1.5"/>
                    <line x1="0" y1="7" x2="100" y2="7" stroke="#C8102E" stroke-width="1"/>
                    <line x1="0" y1="11" x2="80" y2="11" stroke="#C8102E" stroke-width="0.5"/>
                </svg>
                <svg class="f1-car" width="320" height="80" viewBox="0 0 160 40" xmlns="http://www.w3.org/2000/svg">
                    <rect x="130" y="30" width="28" height="4" rx="1" fill="#C8102E"/>
                    <rect x="138" y="26" width="14" height="4" rx="1" fill="#C8102E"/>
                    <rect x="4" y="10" width="18" height="3" rx="1" fill="#C8102E"/>
                    <rect x="9" y="13" width="4" height="10" rx="1" fill="#888"/>
                    <ellipse cx="80" cy="26" rx="62" ry="9" fill="#C8102E"/>
                    <ellipse cx="72" cy="18" rx="16" ry="7" fill="#1a1a1a"/>
                    <rect x="62" y="15" width="32" height="5" rx="3" fill="#333"/>
                    <polygon points="142,22 158,26 142,30" fill="#a50d25"/>
                    <rect x="50" y="20" width="30" height="6" rx="2" fill="#a50d25"/>
                    <ellipse cx="128" cy="34" rx="7" ry="7" fill="#222"/>
                    <ellipse cx="128" cy="34" rx="4" ry="4" fill="#444"/>
                    <ellipse cx="30" cy="34" rx="8" ry="8" fill="#222"/>
                    <ellipse cx="30" cy="34" rx="5" ry="5" fill="#444"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        if st.button("Enter Dashboard →", use_container_width=True, key="landing_enter_btn"):
            st.session_state["show_landing"] = False
            st.rerun()


# 1. Streamlit UI Setup
st.set_page_config(page_title="F1 Driving Style Clustering", layout="wide")

if "show_landing" not in st.session_state:
    st.session_state["show_landing"] = True
if st.session_state["show_landing"]:
    render_landing_page()
    st.stop()
st.markdown(
    """
    <div class="dashboard-hero">
        <div class="dashboard-hero__eyebrow">Insight-first dashboard</div>
        <h1 class="dashboard-hero__title">F1 Driving Style Clustering</h1>
        <p class="dashboard-hero__subtitle">Micro-sector telemetry analysis using FastF1 and unsupervised learning</p>
        <p class="dashboard-hero__description">Compare how drivers brake, carry speed, and return to throttle within a selected distance window.</p>
        <div class="dashboard-hero__accent"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def load_css(filepath="style.css"):
    with open(filepath, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown("""
<script>
(function() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes chartFadeIn {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .chart-animate {
            animation: chartFadeIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
    `;
    document.head.appendChild(style);

    const seen = new WeakSet();

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType !== 1) return;
                const charts = node.matches('[data-testid="stPlotlyChart"]')
                    ? [node]
                    : Array.from(node.querySelectorAll('[data-testid="stPlotlyChart"]'));
                charts.forEach((chart) => {
                    if (seen.has(chart)) return;
                    seen.add(chart);
                    chart.classList.remove('chart-animate');
                    void chart.offsetWidth;
                    chart.classList.add('chart-animate');
                });
            });
        });
    });

    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


def apply_chart_animation(fig):
    fig.update_layout(
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


def section_title(title, subtitle=None):
    subtitle_html = (
        f'<p class="section-title__subtitle">{html.escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div class="section-title">
            <h2 class="section-title__title">{html.escape(title)}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_note(text):
    st.markdown(
        f"""
        <div class="surface-note surface-note--info">
            <div class="surface-note__label">Insight</div>
            <div class="surface-note__body">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_note(text):
    st.markdown(
        f"""
        <div class="surface-note surface-note--warning">
            <div class="surface-note__label">Watch</div>
            <div class="surface-note__body">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_filename_component(value):
    if pd.isna(value):
        return "unknown"
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = ascii_text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def build_export_filename(prefix, analysis_config):
    year = clean_filename_component(analysis_config.get("year"))
    grand_prix = clean_filename_component(analysis_config.get("grand_prix"))
    session = clean_filename_component(analysis_config.get("session_type"))
    start_distance = analysis_config.get("start_distance")
    end_distance = analysis_config.get("end_distance")
    return f"f1_{prefix}_{year}_{grand_prix}_{session}_{start_distance}_{end_distance}.csv"


def build_sample_label(row):
    driver = row.get("Driver", "Unknown")
    sector_name = row.get("Sector_Name", "Selected sector")
    start = row.get("Sector_Start", None)
    end = row.get("Sector_End", None)

    if pd.notna(start) and pd.notna(end):
        return f"{driver} — {sector_name} ({int(start)}–{int(end)} m)"
    return f"{driver} — {sector_name}"


def build_sector_label(row):
    sector_name = row.get("Sector_Name", "Selected sector")
    start = row.get("Sector_Start", None)
    end = row.get("Sector_End", None)

    if pd.notna(start) and pd.notna(end):
        return f"{sector_name} ({int(start)}–{int(end)} m)"
    return str(sector_name)


with st.expander("How to use this dashboard"):
    st.markdown(
        """
        1. Select the race, session, and driver set from the sidebar.
        2. Choose the micro-sector distance window to analyze.
        3. Run the analysis to build features and clusters.
        4. Use Braking Zone Finder if the metrics look flat or hard to interpret.
        5. Treat cluster profiles as exploratory patterns, not absolute driver labels.
        """
    )
enable_fastf1_cache(CACHE_DIR)

if "show_landing" not in st.session_state:
    st.session_state["show_landing"] = True

if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False

if "df" not in st.session_state:
    st.session_state["df"] = None

if "telemetry_data" not in st.session_state:
    st.session_state["telemetry_data"] = None

if "skipped_drivers" not in st.session_state:
    st.session_state["skipped_drivers"] = []

if "analysis_config" not in st.session_state:
    st.session_state["analysis_config"] = {
        "year": DEFAULT_YEAR,
        "grand_prix": DEFAULT_GP,
        "session_type": DEFAULT_SESSION,
        "driver_codes": DEFAULT_DRIVERS[:10],
        "start_distance": DEFAULT_START_DISTANCE,
        "end_distance": DEFAULT_END_DISTANCE,
        "sector_mode": "Single sector",
        "sector_definitions": [
            {
                "Sector_Name": "Selected sector",
                "Sector_Start": DEFAULT_START_DISTANCE,
                "Sector_End": DEFAULT_END_DISTANCE,
            }
        ],
        "n_clusters": 3,
    }
else:
    st.session_state["analysis_config"].setdefault("n_clusters", 3)
    st.session_state["analysis_config"].setdefault("sector_mode", "Single sector")
    st.session_state["analysis_config"].setdefault(
        "sector_definitions",
        [
            {
                "Sector_Name": "Selected sector",
                "Sector_Start": DEFAULT_START_DISTANCE,
                "Sector_End": DEFAULT_END_DISTANCE,
            }
        ],
    )

if "last_load_time_sec" not in st.session_state:
    st.session_state["last_load_time_sec"] = None

for key, default in [
    ("X_scaled", None),
    ("X_imputed_df", None),
    ("kmeans_model", None),
    ("used_feature_columns", []),
    ("missing_feature_columns", []),
    ("cluster_profile_df", None),
    ("driver_profile_df", None),
    ("lap_mode", "Fastest lap only"),
    ("consistency_raw_df", None),
    ("consistency_summary_df", None),
    ("consistency_skipped_drivers", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_data(show_spinner=False)
def fetch_raw_telemetry_cached(year, grand_prix, session_type, driver_codes_tuple):
    driver_codes = list(driver_codes_tuple)
    telemetry_data, skipped_drivers = fetch_driver_telemetry(
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        driver_codes=driver_codes,
        cache_dir=CACHE_DIR,
    )
    return telemetry_data, skipped_drivers


@st.cache_data(show_spinner=False)
def build_features_cached(
    telemetry_data,
    sector_definitions_tuple,
    year,
    grand_prix,
    session_type,
    sector_mode,
):
    sector_definitions = []
    for sector_name, sector_start, sector_end in sector_definitions_tuple:
        sector_definitions.append(
            {
                "Sector_Name": sector_name,
                "Sector_Start": sector_start,
                "Sector_End": sector_end,
            }
        )
    df = build_feature_dataset(
        telemetry_data,
        sector_definitions=sector_definitions,
        metadata={
            "Year": year,
            "Grand_Prix": grand_prix,
            "Session_Type": session_type,
            "Sector_Mode": sector_mode,
        },
    )
    return df


@st.cache_data(show_spinner=False)
def fetch_top5_telemetry_cached(year, grand_prix, session_type, driver_codes_tuple):
    driver_codes = list(driver_codes_tuple)
    return fetch_driver_top5_telemetry(
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        driver_codes=driver_codes,
        cache_dir=CACHE_DIR,
    )


def prepare_ml_features(df, feature_columns):
    available_feature_columns = [column for column in feature_columns if column in df.columns]
    if not available_feature_columns:
        raise ValueError("No ML feature columns available for clustering.")

    X_raw = df.loc[:, available_feature_columns].apply(pd.to_numeric, errors="coerce")
    all_nan_columns = X_raw.columns[X_raw.isna().all()].tolist()
    if all_nan_columns:
        X_raw = X_raw.copy()
        X_raw.loc[:, all_nan_columns] = 0

    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        imputer = SimpleImputer(strategy="median")

    X_imputed = imputer.fit_transform(X_raw)
    if X_imputed.shape[1] != len(available_feature_columns):
        raise ValueError(
            f"ML preprocessing shape mismatch: imputed shape {X_imputed.shape}, "
            f"feature columns {len(available_feature_columns)}"
        )

    X_imputed_df = pd.DataFrame(
        X_imputed,
        columns=available_feature_columns,
        index=df.index,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)

    return X_scaled, X_imputed_df, imputer, scaler, available_feature_columns


def run_kmeans_clustering(df, X_scaled, n_clusters):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clustered_df = df.copy()
    clustered_df["Style_Cluster"] = kmeans_model.fit_predict(X_scaled)

    if n_clusters == 3:
        cluster_mapping = {
            0: "Heavy Trail-Braker",
            1: "High Apex Speed",
            2: "Early Accelerator",
        }
        clustered_df["Driving_Profile"] = clustered_df["Style_Cluster"].map(cluster_mapping)
    else:
        clustered_df["Driving_Profile"] = clustered_df["Style_Cluster"].map(
            lambda cluster_id: f"Cluster {cluster_id}"
        )

    return clustered_df, kmeans_model


def calculate_clustering_metrics(X_scaled, labels):
    return {
        "Silhouette Score": silhouette_score(X_scaled, labels),
        "Davies-Bouldin Score": davies_bouldin_score(X_scaled, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(X_scaled, labels),
    }


def calculate_elbow_curve(X_scaled, max_k):
    inertias = []
    ks = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_scaled)
        ks.append(k)
        inertias.append(kmeans.inertia_)
    return pd.DataFrame({"k": ks, "inertia": inertias})


def profile_clusters(df):
    cluster_feature_columns = [
        "Entry_Speed",
        "Apex_Speed",
        "Exit_Speed",
        "Speed_Loss",
        "Speed_Recovery",
        "Braking_Pct",
        "Throttle_Pct",
        "Full_Throttle_Pct",
        "Braking_Zone_Length",
        "Trail_Braking_Index",
        "Throttle_Commitment_Index",
        "Corner_Aggression_Score",
        "Smoothness_Index",
    ]
    available_cluster_feature_columns = [
        column for column in cluster_feature_columns if column in df.columns
    ]
    if not available_cluster_feature_columns:
        raise ValueError("No cluster profiling columns are available.")

    group_columns = ["Style_Cluster", "Driving_Profile"]
    if "Sector_Name" in df.columns:
        group_columns.append("Sector_Name")

    cluster_profile_df = (
        df.groupby(group_columns, as_index=False)[available_cluster_feature_columns]
        .mean()
        .round(2)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    if "Sector_Name" in cluster_profile_df.columns:
        cluster_profile_df["Cluster_Sector_Label"] = (
            cluster_profile_df["Driving_Profile"].astype(str)
            + " | "
            + cluster_profile_df["Sector_Name"].astype(str)
        )
    return cluster_profile_df


def generate_cluster_interpretation(cluster_profile_df, cluster_id):
    selected_cluster = cluster_profile_df.loc[
        cluster_profile_df["Style_Cluster"] == cluster_id
    ].iloc[0]
    overall_means = cluster_profile_df.select_dtypes(include="number").mean()

    interpretation_parts = []
    if selected_cluster["Braking_Pct"] > overall_means["Braking_Pct"]:
        interpretation_parts.append(
            "This cluster tends to show heavier braking usage relative to the other clusters in the selected telemetry window."
        )
    if selected_cluster["Apex_Speed"] > overall_means["Apex_Speed"]:
        interpretation_parts.append(
            "This cluster tends to preserve a higher apex speed through the micro-sector, based on the selected telemetry window."
        )
    if selected_cluster["Speed_Loss"] > overall_means["Speed_Loss"]:
        interpretation_parts.append(
            "This cluster tends to show stronger speed loss during the corner, which suggests sharper deceleration in this segment."
        )
    if selected_cluster["Throttle_Commitment_Index"] > overall_means["Throttle_Commitment_Index"]:
        interpretation_parts.append(
            "This cluster tends to show stronger throttle commitment after apex, based on the selected telemetry window."
        )
    if selected_cluster["Smoothness_Index"] > overall_means["Smoothness_Index"]:
        interpretation_parts.append(
            "This cluster tends to show smoother speed trace behavior in this telemetry window."
        )
    if selected_cluster["Corner_Aggression_Score"] > overall_means["Corner_Aggression_Score"]:
        interpretation_parts.append(
            "This cluster tends to show a more aggressive cornering profile, based on the selected telemetry window."
        )

    if not interpretation_parts:
        interpretation_parts.append(
            "Based on the selected telemetry window, this cluster sits around the overall cluster average and does not strongly stand out on the main braking, apex, throttle, or smoothness indicators."
        )

    narrative = " ".join(interpretation_parts)
    return (
        f"{narrative} This suggests a characteristic pattern in this cluster, not a fixed driver label."
    )


def get_mode_or_unknown(series):
    mode_values = series.dropna().mode()
    if len(mode_values) == 0:
        return "Unknown"
    return mode_values.iloc[0]


def aggregate_driver_profiles(df):
    available_profile_feature_columns = [
        column for column in DRIVER_PROFILE_NUMERIC_COLUMNS if column in df.columns and column != "Sample_Count"
    ]
    if not available_profile_feature_columns:
        raise ValueError("No driver aggregation columns are available.")

    driver_means = (
        df.groupby("Driver", as_index=False)[available_profile_feature_columns]
        .mean()
        .round(2)
    )
    sample_counts = df.groupby("Driver").size().reset_index(name="Sample_Count")
    most_common_profile = (
        df.groupby("Driver")["Driving_Profile"]
        .agg(get_mode_or_unknown)
        .reset_index(name="Most_Frequent_Driving_Profile")
    )

    driver_profile_df = driver_means.merge(sample_counts, on="Driver", how="left")
    driver_profile_df = driver_profile_df.merge(
        most_common_profile,
        on="Driver",
        how="left",
    )

    for col in available_profile_feature_columns + ["Sample_Count"]:
        if col in driver_profile_df.columns:
            driver_profile_df[col] = pd.to_numeric(driver_profile_df[col], errors="coerce")

    driver_profile_df = driver_profile_df.sort_values("Driver").reset_index(drop=True)
    return driver_profile_df


def format_metric_value(value):
    if value is None:
        return "Not detected"
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "Not detected"
    try:
        numeric_value = float(value)
        return round(numeric_value, 2)
    except (TypeError, ValueError):
        return value


def calculate_metric_difference(value1, value2):
    if pd.isna(value1) or pd.isna(value2):
        return "N/A"
    if isinstance(value1, str) or isinstance(value2, str):
        if value1 == value2:
            return "Same"
        return "Different"
    try:
        numeric_difference = round(float(value1) - float(value2), 2)
        return numeric_difference
    except (TypeError, ValueError):
        return "N/A"


def detect_braking_zones(telemetry_df):
    if telemetry_df is None or telemetry_df.empty:
        return pd.DataFrame(columns=[
            "Brake_Start_Distance",
            "Brake_End_Distance",
            "Zone_Length",
            "Entry_Speed",
            "Minimum_Speed",
            "Speed_Loss",
        ])

    required_columns = ["Distance", "Speed", "Brake"]
    missing_columns = [column for column in required_columns if column not in telemetry_df.columns]
    if missing_columns:
        return pd.DataFrame(columns=[
            "Brake_Start_Distance",
            "Brake_End_Distance",
            "Zone_Length",
            "Entry_Speed",
            "Minimum_Speed",
            "Speed_Loss",
        ])

    zone_df = telemetry_df[["Distance", "Speed", "Brake"]].copy()
    zone_df["Brake_Active"] = pd.to_numeric(zone_df["Brake"], errors="coerce").fillna(0) > 0
    zone_df["Distance"] = pd.to_numeric(zone_df["Distance"], errors="coerce")
    zone_df["Speed"] = pd.to_numeric(zone_df["Speed"], errors="coerce")
    zone_df = zone_df.dropna(subset=["Distance", "Speed"])

    transition_mask = zone_df["Brake_Active"] != zone_df["Brake_Active"].shift(fill_value=False)
    zone_df["Zone_ID"] = transition_mask.cumsum()
    braking_segments = zone_df[zone_df["Brake_Active"]].groupby("Zone_ID")

    zone_rows = []
    for _, segment in braking_segments:
        if segment.empty:
            continue
        start_distance = float(segment["Distance"].iloc[0])
        end_distance = float(segment["Distance"].iloc[-1])
        entry_speed = float(segment["Speed"].iloc[0])
        minimum_speed = float(segment["Speed"].min())
        zone_rows.append({
            "Brake_Start_Distance": start_distance,
            "Brake_End_Distance": end_distance,
            "Zone_Length": round(end_distance - start_distance, 2),
            "Entry_Speed": round(entry_speed, 2),
            "Minimum_Speed": round(minimum_speed, 2),
            "Speed_Loss": round(entry_speed - minimum_speed, 2),
        })

    if not zone_rows:
        return pd.DataFrame(columns=[
            "Brake_Start_Distance",
            "Brake_End_Distance",
            "Zone_Length",
            "Entry_Speed",
            "Minimum_Speed",
            "Speed_Loss",
        ])

    zone_df = pd.DataFrame(zone_rows).sort_values("Speed_Loss", ascending=False).reset_index(drop=True)
    return zone_df


def render_cluster_profiles(df):
    render_section_header(
        "Cluster Profiles",
        "Explain each cluster using averaged telemetry features and representative samples.",
    )
    info_note("This section explains what each cluster tends to represent using average telemetry features.")
    if df is not None and "Corner_Aggression_Score" in df.columns:
        st.caption("Aggression score is a composite index from normalized speed loss, braking involvement, and throttle commitment. It is not an official F1 metric.")
    render_section_guidance(
        "These profiles turn cluster assignments into readable driving-style patterns.",
        "Read the chart first for the overall cluster shape, then open the cluster detail panels for representative examples.",
        "Some clusters may overlap when the selected sector is too flat or the driver set is very narrow.",
    )

    if df is None or "Style_Cluster" not in df.columns or "Driving_Profile" not in df.columns:
        st.info("Run Micro-Sector Analysis to populate cluster profiles.")
        return

    try:
        cluster_profile_df = profile_clusters(df)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.session_state["cluster_profile_df"] = cluster_profile_df

    chart_columns = [
        "Apex_Speed",
        "Braking_Pct",
        "Throttle_Commitment_Index",
        "Corner_Aggression_Score",
    ]
    available_chart_columns = [column for column in chart_columns if column in cluster_profile_df.columns]
    if available_chart_columns:
        chart_metric = available_chart_columns[0]
        chart_label = prettify_label(chart_metric)
        chart_x_values = cluster_profile_df["Cluster_Sector_Label"] if "Cluster_Sector_Label" in cluster_profile_df.columns else cluster_profile_df["Driving_Profile"]
        fig = go.Figure(go.Bar(
            x=list(chart_x_values),
            y=list(cluster_profile_df[chart_metric]),
            marker_color="#1d4ed8",
        ))
        fig.update_layout(
            title=f"Average {chart_label} by cluster and sector",
            xaxis_title="Driving profile / sector",
            yaxis_title=chart_label,
        )
        apply_chart_animation(fig)
        st.plotly_chart(fig, use_container_width=True)
        if chart_metric == "Apex_Speed":
            st.caption("Higher Apex Speed means the driver carried more speed at the slowest point of the selected window.")
        elif chart_metric == "Braking_Pct":
            st.caption("Higher Braking % means the driver spent more of the selected window braking.")
        elif chart_metric == "Throttle_Commitment_Index":
            st.caption("Throttle commitment describes how strongly the driver returned to throttle after the apex.")

    with st.expander("View cluster summary table"):
        st.dataframe(cluster_profile_df, use_container_width=True, hide_index=True)

    for _, cluster_row in cluster_profile_df.iterrows():
        cluster_id = int(cluster_row["Style_Cluster"])
        cluster_label = str(cluster_row["Driving_Profile"])
        with st.expander(f"Cluster {cluster_id} — {cluster_label}"):
            st.markdown(generate_cluster_interpretation(cluster_profile_df, cluster_id))

            sample_columns = ["Driver"]
            if "Sector_Name" in df.columns:
                sample_columns.extend(["Sector_Name", "Sector_Start", "Sector_End"])
            if "Grand_Prix" in df.columns:
                sample_columns.append("Grand_Prix")
            if "Session_Type" in df.columns:
                sample_columns.append("Session_Type")
            sample_columns.extend([
                "Entry_Speed",
                "Apex_Speed",
                "Braking_Pct",
                "Throttle_Pct",
                "Corner_Aggression_Score",
            ])
            sample_columns = [column for column in sample_columns if column in df.columns]

            cluster_samples = df.loc[df["Style_Cluster"] == cluster_id, sample_columns].copy()
            if cluster_samples.empty:
                st.info("No sample rows are available for this cluster.")
            else:
                st.dataframe(cluster_samples.round(2), use_container_width=True, hide_index=True)

    with st.expander("View cluster assignments"):
        assignment_df = df[["Driver", "Style_Cluster", "Driving_Profile"]].copy()
        st.dataframe(assignment_df.sort_values(["Style_Cluster", "Driver"]), use_container_width=True, hide_index=True)


def safe_display_value(value, decimals=2):
    if pd.isna(value):
        return "Not detected"
    if isinstance(value, (int, float, np.number)):
        return round(float(value), decimals)
    return value


def generate_driver_profile_interpretation(
    selected_driver_row,
    dataset_averages,
    sample_count,
    sector_quality_warning=False,
):
    driver_label = selected_driver_row.get("Driver")
    if pd.isna(driver_label):
        driver_label = "The selected driver"

    observations = []
    observation_lookup = {
        "Braking_Pct": (
            "above-average braking usage",
            "below-average braking usage",
            "Braking_Pct",
        ),
        "Apex_Speed": (
            "higher-than-average apex speed",
            "lower-than-average apex speed",
            "Apex_Speed",
        ),
        "Throttle_Commitment_Index": (
            "stronger throttle commitment",
            "weaker throttle commitment",
            "Throttle_Commitment_Index",
        ),
        "Corner_Aggression_Score": (
            "higher aggression score",
            "lower aggression score",
            "Corner_Aggression_Score",
        ),
    }

    for metric, (above_phrase, below_phrase, column_name) in observation_lookup.items():
        if metric not in selected_driver_row.index or metric not in dataset_averages.index:
            continue
        driver_value = selected_driver_row[metric]
        average_value = dataset_averages[metric]
        if pd.isna(driver_value) or pd.isna(average_value):
            continue
        if driver_value > average_value:
            observations.append(above_phrase)
        else:
            observations.append(below_phrase)

    if not observations:
        return "Not enough valid telemetry metrics are available to generate a reliable driver interpretation for this selected sector."

    if len(observations) == 1:
        observation_sentence = (
            f"Within this selected micro-sector, {driver_label} shows {observations[0]} compared with the loaded driver set."
        )
    elif len(observations) == 2:
        observation_sentence = (
            f"Within this selected micro-sector, {driver_label} shows {observations[0]} and {observations[1]} compared with the loaded driver set."
        )
    else:
        observation_sentence = (
            f"Within this selected micro-sector, {driver_label} shows {', '.join(observations[:-1])}, and {observations[-1]} compared with the loaded driver set."
        )

    interpretation_sentences = [observation_sentence]

    braking_pct = selected_driver_row.get("Braking_Pct")
    apex_speed = selected_driver_row.get("Apex_Speed")
    throttle_commitment = selected_driver_row.get("Throttle_Commitment_Index")

    if pd.notna(braking_pct) and pd.notna(apex_speed):
        avg_braking_pct = dataset_averages.get("Braking_Pct")
        avg_apex_speed = dataset_averages.get("Apex_Speed")
        if pd.notna(avg_braking_pct) and pd.notna(avg_apex_speed):
            if braking_pct > avg_braking_pct and apex_speed > avg_apex_speed:
                interpretation_sentences.append(
                    "That points to a controlled, committed cornering profile with braking still present while carrying strong minimum speed."
                )
            elif braking_pct > avg_braking_pct and apex_speed <= avg_apex_speed:
                interpretation_sentences.append(
                    "That points to a heavier braking profile, where speed is given up before or around the apex."
                )
            elif braking_pct <= avg_braking_pct and apex_speed > avg_apex_speed:
                interpretation_sentences.append(
                    "That points to a smoother high-minimum-speed profile with less braking inside the selected window."
                )

    if pd.notna(throttle_commitment):
        avg_throttle_commitment = dataset_averages.get("Throttle_Commitment_Index")
        if pd.notna(avg_throttle_commitment):
            if throttle_commitment > avg_throttle_commitment:
                interpretation_sentences.append(
                    "The stronger throttle commitment suggests earlier or stronger acceleration through the exit phase."
                )
            else:
                interpretation_sentences.append(
                    "The weaker throttle commitment suggests a more conservative throttle application after the apex."
                )

    reliability_notes = []
    if int(sample_count) == 1:
        reliability_notes.append(
            "Because this is based on one loaded sample, treat it as a session-specific observation rather than a stable driver trait."
        )
    if sector_quality_warning:
        reliability_notes.append(
            "Because this sector has limited braking or speed-loss variation, style conclusions are less reliable."
        )
    if reliability_notes:
        interpretation_sentences.append(" ".join(reliability_notes))

    return " ".join(interpretation_sentences)


def render_driver_profiles(df):
    render_section_header(
        "Driver Profiles",
        "Summarize average driver behavior across the currently loaded samples.",
    )
    info_note("This section summarizes each selected driver's behavior across the currently loaded samples.")
    if "Sector_Name" in df.columns and df["Sector_Name"].nunique() > 1:
        st.caption("Driver profiles aggregate across all active sectors in this run. Use Sector Comparison to inspect sector-specific differences.")
    render_section_guidance(
        "This section compares the aggregated behavior of each driver inside the selected micro-sector.",
        "Use the driver comparison chart first, then inspect the selected driver summary and interpretation for the narrative.",
        "Very flat sectors can make drivers look similar, so a weak warning is expected when braking and speed-loss variation are low.",
    )

    if df is None or "Driver" not in df.columns:
        st.info("Run Micro-Sector Analysis to populate driver profiles.")
        return

    try:
        driver_profile_df = aggregate_driver_profiles(df)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.session_state["driver_profile_df"] = driver_profile_df

    driver_profile_numeric_cols = [
        col for col in DRIVER_PROFILE_NUMERIC_COLUMNS if col in driver_profile_df.columns
    ]
    for col in driver_profile_numeric_cols:
        driver_profile_df[col] = pd.to_numeric(driver_profile_df[col], errors="coerce")

    avg_speed_loss = float(pd.to_numeric(df["Speed_Loss"], errors="coerce").mean())
    avg_braking_pct = float(pd.to_numeric(df["Braking_Pct"], errors="coerce").mean())
    throttle_pct_mean = float(pd.to_numeric(df["Throttle_Pct"], errors="coerce").mean())
    braking_detected_count = int((pd.to_numeric(df["Braking_Pct"], errors="coerce") > 0).sum())
    total_samples = int(len(df))

    st.markdown("**Micro-sector quality check**")
    render_metric_cards(
        [
            ("Average speed loss", round(avg_speed_loss, 2), "Average speed drop across the selected window."),
            ("Average braking %", round(avg_braking_pct, 2), "Average braking share inside the selected window."),
            ("Average throttle %", round(throttle_pct_mean, 2), "Average throttle usage across the selected samples."),
            ("Braking-detected samples", braking_detected_count, "Samples with braking detected in the loaded telemetry."),
            ("Total samples", total_samples, "Total samples represented in the current driver aggregation."),
        ]
    )

    if avg_speed_loss < 5 and avg_braking_pct < 5:
        st.warning(
            "This selected micro-sector appears to contain little braking or speed loss. Driver profiles may be weak or flat. Try using the Braking Zone Finder or select a wider braking-heavy sector."
        )

    if (pd.to_numeric(df["Throttle_Pct"], errors="coerce") >= 95).mean() > 0.5:
        st.warning(
            "Most drivers are at high throttle through this window, so throttle-based metrics may not separate styles well."
        )

    preferred_metric_order = [
        "Corner_Aggression_Score",
        "Apex_Speed",
        "Braking_Pct",
        "Throttle_Commitment_Index",
        "Smoothness_Index",
        "Speed_Loss",
        "Sample_Count",
    ]
    metric_options = [
        metric for metric in preferred_metric_order if metric in driver_profile_numeric_cols
    ]
    if not metric_options:
        st.warning("No numeric metrics are available for driver profile sorting.")
        return

    metric_display_options = [prettify_label(metric) for metric in metric_options]
    selected_metric_display = st.selectbox(
        "Sort drivers by",
        options=metric_display_options,
        key="driver_profile_sort_metric_display",
    )
    selected_metric = metric_options[metric_display_options.index(selected_metric_display)]

    if driver_profile_df[selected_metric].dropna().empty:
        st.warning("The selected metric has no valid values. Switching to the first available numeric metric.")
        selected_metric = metric_options[0]

    sorted_driver_profile_df = driver_profile_df.sort_values(
        by=selected_metric,
        ascending=False,
    ).reset_index(drop=True)

    metric_range = float(sorted_driver_profile_df[selected_metric].max() - sorted_driver_profile_df[selected_metric].min())
    metric_std = float(sorted_driver_profile_df[selected_metric].std())
    if metric_range < 1.0 or metric_std < 0.5:
        st.warning(
            "The selected metric has very low variation across drivers in this micro-sector. Try a braking-heavy sector or use Braking Zone Finder."
        )

    render_metric_cards(
        [
            ("Selected metric", prettify_label(selected_metric), "Metric currently used to rank drivers."),
            ("Top driver", str(sorted_driver_profile_df.iloc[0]["Driver"]), "Highest-ranked driver in the current comparison."),
            ("Sample coverage", int(sorted_driver_profile_df["Sample_Count"].sum()), "Combined driver samples represented in the profile summary."),
        ]
    )
    st.caption(f"Selected metric: {prettify_label(selected_metric)}")

    chart_df = sorted_driver_profile_df.dropna(subset=[selected_metric]).copy()
    chart_df = chart_df.sort_values(by=selected_metric, ascending=True).reset_index(drop=True)
    fig = px.bar(
        chart_df,
        x=selected_metric,
        y="Driver",
        orientation="h",
        text=chart_df[selected_metric].round(2),
        labels={selected_metric: prettify_label(selected_metric), "Driver": "Driver"},
        title=f"Driver comparison by {prettify_label(selected_metric)}",
        color_discrete_sequence=["#1d4ed8"],
    )
    fig.update_traces(textposition="outside")
    apply_chart_animation(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("If all bars look similar, the selected metric may not separate drivers strongly in this micro-sector.")

    if selected_metric == "Apex_Speed":
        st.caption("Higher Apex Speed means the driver carried more speed at the slowest point of the selected window.")
    elif selected_metric == "Braking_Pct":
        st.caption("Higher Braking % means the driver spent more of the selected window braking.")
    elif selected_metric == "Throttle_Commitment_Index":
        st.caption("Throttle commitment describes how strongly the driver returned to throttle after the apex.")
    elif selected_metric == "Corner_Aggression_Score":
        st.caption("Aggression score is a composite index from normalized speed loss, braking involvement, and throttle commitment. It is not an official F1 metric.")
    else:
        st.caption("This chart compares the selected metric across drivers in the active micro-sector.")

    with st.expander("View driver profile table"):
        st.dataframe(sorted_driver_profile_df, use_container_width=True, hide_index=True)

    available_drivers = sorted_driver_profile_df["Driver"].tolist()
    selected_driver = st.selectbox(
        "Select driver",
        options=available_drivers,
        key="driver_profile_selected_driver",
    )

    selected_driver_row = sorted_driver_profile_df.loc[
        sorted_driver_profile_df["Driver"] == selected_driver
    ].iloc[0]

    st.caption("Not detected usually means the selected window did not contain braking or throttle reapplication for that driver.")
    st.markdown("**Selected Driver Summary**")
    metric_columns = st.columns(3)
    metric_summary_rows = [
        ("Sample count", int(selected_driver_row["Sample_Count"])),
        ("Most frequent driving profile", selected_driver_row["Most_Frequent_Driving_Profile"]),
        ("Avg Apex Speed", selected_driver_row["Apex_Speed"]),
        ("Avg Braking %", selected_driver_row["Braking_Pct"]),
        ("Avg Throttle Commitment", selected_driver_row["Throttle_Commitment_Index"]),
        ("Avg Aggression Score", selected_driver_row["Corner_Aggression_Score"]),
    ]
    for index, (label, value) in enumerate(metric_summary_rows):
        col = metric_columns[index % 3]
        col.metric(label, safe_display_value(value))

    numeric_profile_df = driver_profile_df.select_dtypes(include=["number"])
    dataset_averages = numeric_profile_df.mean(numeric_only=True)
    driver_interpretation = generate_driver_profile_interpretation(
        selected_driver_row,
        dataset_averages,
        int(selected_driver_row["Sample_Count"]),
        sector_quality_warning=bool(avg_speed_loss < 5 and avg_braking_pct < 5),
    )

    st.markdown("**Interpretation**")
    st.markdown(driver_interpretation)

    if "Grand_Prix" in df.columns and df["Grand_Prix"].nunique() <= 1:
        st.info("Aggregation is based on the currently loaded samples. For stronger driver-level conclusions, use multiple races.")


def render_model_evaluation(df, X_scaled):
    render_section_header(
        "Model Evaluation",
        "Evaluate clustering separation, inertia, and cluster balance for the current run.",
    )
    info_note("These metrics check whether the unsupervised clusters are reasonably separated. They are not accuracy scores because there is no labelled target.")
    render_section_guidance(
        "This section shows whether the cluster structure is meaningful or whether the window needs better separation.",
        "Check the key metric cards first, then use the elbow curve to understand the cluster count choice.",
        "A sparse or flat sector can produce weak separation and unstable cluster metrics.",
    )

    if df is None or X_scaled is None or "Style_Cluster" not in df.columns:
        st.info("Run Micro-Sector Analysis to populate model evaluation details.")
        return

    used_feature_columns = st.session_state.get("used_feature_columns") or []
    missing_feature_columns = st.session_state.get("missing_feature_columns") or []
    if used_feature_columns:
        st.caption(
            f"ML used {len(used_feature_columns)} available features out of {len(CLUSTER_FEATURE_COLUMNS)} requested features."
        )
    if missing_feature_columns:
        st.info(
            "Missing optional features were skipped: " + ", ".join(missing_feature_columns)
        )

    labels = df["Style_Cluster"].to_numpy()
    n_samples = len(df)
    n_clusters = int(df["Style_Cluster"].nunique())
    metrics_available = n_samples > n_clusters and n_clusters > 1

    st.markdown(f"**Current clustering setup:** {n_clusters} clusters across {n_samples} samples.")

    if metrics_available:
        metrics = calculate_clustering_metrics(X_scaled, labels)
        render_metric_cards(
            [
                ("Silhouette", round(float(metrics["Silhouette Score"]), 3), "Higher values indicate clearer cluster separation."),
                ("Davies-Bouldin", round(float(metrics["Davies-Bouldin Score"]), 3), "Lower values indicate better separation."),
                ("Calinski-Harabasz", round(float(metrics["Calinski-Harabasz Score"]), 3), "Higher values indicate stronger cluster structure."),
            ]
        )
    else:
        st.warning("Not enough samples to compute reliable clustering metrics.")

    if metrics_available:
        max_k_elbow = min(8, max(2, n_samples - 1))
        elbow_curve = calculate_elbow_curve(X_scaled, max_k_elbow)
        if not elbow_curve.empty:
            sil_scores_per_k = []
            for k_val in range(2, max_k_elbow + 1):
                km_tmp = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
                labels_tmp = km_tmp.fit_predict(X_scaled)
                if len(set(labels_tmp)) > 1:
                    sil_scores_per_k.append(round(float(silhouette_score(X_scaled, labels_tmp)), 4))
                else:
                    sil_scores_per_k.append(None)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=elbow_curve["k"],
                y=elbow_curve["inertia"],
                mode="lines+markers",
                name="Inertia",
                yaxis="y1",
            ))
            fig.add_trace(go.Scatter(
                x=list(range(2, max_k_elbow + 1)),
                y=sil_scores_per_k,
                mode="lines+markers",
                name="Silhouette",
                yaxis="y2",
            ))
            fig.update_layout(
                title="Elbow Method: Inertia and Silhouette vs Number of Clusters",
                xaxis=dict(title="Number of clusters (k)"),
                yaxis=dict(title="Inertia"),
                yaxis2=dict(title="Silhouette Score", overlaying="y", side="right"),
                legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
            )
            apply_chart_animation(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Inertia (left axis) drops as k rises; look for the elbow. Silhouette (right axis) peaks at the k with best-separated clusters.")

    cluster_distribution = (
        df.groupby(["Style_Cluster", "Driving_Profile"])
        .size()
        .reset_index(name="Count")
        .sort_values("Style_Cluster")
        .reset_index(drop=True)
    )

    with st.expander("View clustering metrics"):
        if metrics_available:
            metrics_df = pd.DataFrame(
                [
                    {"Metric": metric_name, "Value": value}
                    for metric_name, value in metrics.items()
                ]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("Clustering metrics are unavailable for the current sample setup.")

    with st.expander("View cluster distribution"):
        st.dataframe(cluster_distribution, use_container_width=True, hide_index=True)

    st.markdown("**Interpretation**")
    st.markdown(
        "- Higher Silhouette Score is better.\n"
        "- Lower Davies-Bouldin Score is better.\n"
        "- Higher Calinski-Harabasz Score is better.\n"
        "- The elbow curve helps inspect a reasonable k value."
    )


def render_sector_comparison(df):
    render_section_header(
        "Sector Comparison",
        "Compare how the same drivers behave across multiple selected distance windows.",
    )
    info_note("Compare how the same drivers behave across multiple selected distance windows.")

    if df is None or "Sector_Name" not in df.columns:
        st.warning("Run a multi-sector analysis to populate sector comparison data.")
        return

    metric_options = [
        "Apex_Speed",
        "Speed_Loss",
        "Braking_Pct",
        "Throttle_Commitment_Index",
        "Corner_Aggression_Score",
        "Smoothness_Index",
    ]
    selected_metric = st.selectbox(
        "Choose metric",
        options=metric_options,
        key="sector_comparison_metric",
    )

    sector_order = df["Sector_Name"].drop_duplicates().tolist()
    comparison_pivot = (
        df.pivot_table(
            index="Driver",
            columns="Sector_Name",
            values=selected_metric,
            aggfunc="mean",
        )
        .reindex(columns=sector_order)
        .round(2)
    )
    # reset_index() brings Driver back as a regular column;
    # rename_axis(None, axis=1) clears the "Sector_Name" columns.name label
    # that pivot_table sets, which Streamlit would otherwise render as a
    # spurious sub-header row above the real column names.
    display_pivot = comparison_pivot.reset_index().rename_axis(None, axis=1)
    st.dataframe(display_pivot, use_container_width=True, hide_index=True)

    plot_df = comparison_pivot.reset_index().melt(
        id_vars="Driver",
        var_name="Sector_Name",
        value_name=selected_metric,
    )
    metric_label = prettify_label(selected_metric)
    fig = px.bar(
        plot_df,
        x="Sector_Name",
        y=selected_metric,
        color="Driver",
        barmode="group",
        labels={selected_metric: metric_label, "Sector_Name": "Sector"},
        hover_data=["Driver", "Sector_Name", selected_metric],
    )
    fig.update_layout(title=f"{metric_label} by driver and sector")
    apply_chart_animation(fig)
    st.plotly_chart(fig, use_container_width=True)

    highest_braking_sector = (
        df.groupby("Sector_Name")["Braking_Pct"].mean().sort_values(ascending=False).idxmax()
    )
    highest_speed_loss_sector = (
        df.groupby("Sector_Name")["Speed_Loss"].mean().sort_values(ascending=False).idxmax()
    )
    highest_aggression_driver = (
        df.groupby("Driver")["Corner_Aggression_Score"].mean().sort_values(ascending=False).idxmax()
    )
    highest_braking_value = round(
        df.groupby("Sector_Name")["Braking_Pct"].mean().sort_values(ascending=False).iloc[0],
        2,
    )
    highest_speed_loss_value = round(
        df.groupby("Sector_Name")["Speed_Loss"].mean().sort_values(ascending=False).iloc[0],
        2,
    )
    highest_aggression_value = round(
        df.groupby("Driver")["Corner_Aggression_Score"].mean().sort_values(ascending=False).iloc[0],
        2,
    )

    st.markdown("**Auto insight**")
    st.markdown(
        f"- Highest average braking %: {highest_braking_sector} ({highest_braking_value}%).\n"
        f"- Highest average speed loss: {highest_speed_loss_sector} ({highest_speed_loss_value} km/h).\n"
        f"- Highest average aggression score: {highest_aggression_driver} ({highest_aggression_value})."
    )


def render_dataset_export():
    render_section_header(
        "Dataset Export",
        "Download the stored analysis products for external review and modeling.",
    )
    info_note("Download the engineered telemetry dataset and ML-ready feature matrix for offline analysis.")

    if not st.session_state.get("analysis_ready") or st.session_state.get("df") is None:
        st.warning("Run an analysis first to enable dataset export.")
        return

    df = st.session_state.get("df")
    x_imputed_df = st.session_state.get("X_imputed_df")
    cluster_profile_df = st.session_state.get("cluster_profile_df")
    driver_profile_df = st.session_state.get("driver_profile_df")
    analysis_config = st.session_state.get("analysis_config") or {}

    driver_codes = analysis_config.get("driver_codes") or []
    driver_names = ", ".join(driver_codes) if driver_codes else "None"
    sample_count = len(df)
    start_distance = analysis_config.get("start_distance")
    end_distance = analysis_config.get("end_distance")

    st.markdown("**Active export configuration**")
    export_summary = pd.DataFrame(
        {
            "Field": [
                "Year",
                "Grand Prix",
                "Session",
                "Drivers",
                "Distance window",
                "Number of samples",
            ],
            "Value": [
                analysis_config.get("year"),
                analysis_config.get("grand_prix"),
                analysis_config.get("session_type"),
                f"{driver_names} ({len(driver_codes)})",
                f"{start_distance}-{end_distance} m",
                sample_count,
            ],
        }
    )
    st.dataframe(export_summary, use_container_width=True, hide_index=True)

    engineered_bytes = df.to_csv(index=False).encode("utf-8")
    engineered_filename = build_export_filename("features", analysis_config)
    st.download_button(
        label="Download engineered feature dataset (.csv)",
        data=engineered_bytes,
        file_name=engineered_filename,
        mime="text/csv",
        key="download_engineered_feature_dataset",
    )

    if isinstance(x_imputed_df, pd.DataFrame):
        x_imputed_bytes = x_imputed_df.to_csv(index=False).encode("utf-8")
        ml_filename = build_export_filename("ml_features", analysis_config)
        st.download_button(
            label="Download ML feature matrix (.csv)",
            data=x_imputed_bytes,
            file_name=ml_filename,
            mime="text/csv",
            key="download_ml_feature_matrix",
        )
    else:
        st.info("ML feature matrix is not currently available for export.")

    if isinstance(cluster_profile_df, pd.DataFrame) and not cluster_profile_df.empty:
        cluster_profile_bytes = cluster_profile_df.to_csv(index=False).encode("utf-8")
        cluster_profile_filename = build_export_filename("cluster_profiles", analysis_config)
        st.download_button(
            label="Download cluster profile summary (.csv)",
            data=cluster_profile_bytes,
            file_name=cluster_profile_filename,
            mime="text/csv",
            key="download_cluster_profile_summary",
        )
    else:
        st.caption("Cluster profile summary is not available until cluster profiles have been generated.")

    if isinstance(driver_profile_df, pd.DataFrame) and not driver_profile_df.empty:
        driver_profile_bytes = driver_profile_df.to_csv(index=False).encode("utf-8")
        driver_profile_filename = build_export_filename("driver_profiles", analysis_config)
        st.download_button(
            label="Download driver profile summary (.csv)",
            data=driver_profile_bytes,
            file_name=driver_profile_filename,
            mime="text/csv",
            key="download_driver_profile_summary",
        )
    else:
        st.caption("Driver profile summary is not available until driver profiles have been generated.")


def get_driver_telemetry(telemetry_data, driver):
    if telemetry_data is None:
        return None

    if driver in telemetry_data:
        return telemetry_data[driver]

    for key, tel in telemetry_data.items():
        key_str = str(key)
        if key_str.startswith(f"{driver}__") or key_str.endswith(f"_{driver}") or driver in key_str.split("__"):
            return tel

    return None


def plot_telemetry_traces(trace_specs, column, title, ylabel):
    valid_traces = []
    skipped_labels = []

    for spec in trace_specs:
        tel = spec.get("telemetry")
        if tel is None or (hasattr(tel, "empty") and tel.empty):
            skipped_labels.append(spec["label"])
            continue
        if "Distance" not in tel.columns:
            skipped_labels.append(spec["label"])
            continue

        sector_start = spec.get("sector_start")
        sector_end = spec.get("sector_end")
        try:
            if sector_start is not None and sector_end is not None and pd.notna(sector_start) and pd.notna(sector_end):
                s_start = float(sector_start)
                s_end = float(sector_end)
                sector_tel = tel[
                    (tel["Distance"].astype(float) >= s_start) &
                    (tel["Distance"].astype(float) <= s_end)
                ].copy()
            else:
                sector_tel = tel.copy()
        except Exception:
            skipped_labels.append(spec["label"])
            continue

        if sector_tel.empty:
            skipped_labels.append(spec["label"])
            continue

        if column not in sector_tel.columns:
            skipped_labels.append(spec["label"])
            continue

        valid_traces.append({
            "label": spec["label"],
            "sector_tel": sector_tel,
        })

    if not valid_traces:
        return False, skipped_labels

    fig = go.Figure()
    for trace in valid_traces:
        sector_tel = trace["sector_tel"]
        y_vals = sector_tel[column]
        if hasattr(y_vals, "astype"):
            try:
                y_vals = y_vals.astype(float)
            except Exception:
                pass
        fig.add_trace(go.Scatter(
            x=sector_tel["Distance"].astype(float),
            y=y_vals,
            mode="lines",
            name=str(trace["label"]),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Distance (m)",
        yaxis_title=ylabel,
    )
    return fig, skipped_labels


def get_driver_row(df, driver_code):
    match = df[df['Driver'] == driver_code]
    if match.empty:
        return None
    return match.iloc[0]


def build_driver_comparison(df, driver1, driver2):
    row1 = get_driver_row(df, driver1)
    row2 = get_driver_row(df, driver2)
    if row1 is None or row2 is None:
        return None

    metric_specs = [
        ('Entry Speed', 'Entry_Speed', True),
        ('Apex Speed', 'Apex_Speed', True),
        ('Braking %', 'Braking_Pct', True),
        ('Throttle %', 'Throttle_Pct', True),
        ('Driving Profile', 'Driving_Profile', False),
    ]
    metric_specs = [spec for spec in metric_specs if spec[1] in df.columns]

    rows = []
    for label, column, is_numeric in metric_specs:
        val1 = row1[column]
        val2 = row2[column]
        if is_numeric:
            driver1_val = val1
            driver2_val = val2
            difference = calculate_metric_difference(val1, val2)
        else:
            driver1_val = val1
            driver2_val = val2
            difference = 'Same' if val1 == val2 else 'Different'

        rows.append({
            'Metric': label,
            'Driver 1 value': driver1_val,
            'Driver 2 value': driver2_val,
            'Difference': difference,
        })

    return pd.DataFrame(rows)

def build_driver_comparison_advanced(df, driver1, driver2):
    row1 = get_driver_row(df, driver1)
    row2 = get_driver_row(df, driver2)
    if row1 is None or row2 is None:
        return None

    advanced_metrics = [
        ("Exit Speed", "Exit_Speed"),
        ("Speed Loss", "Speed_Loss"),
        ("Speed Recovery", "Speed_Recovery"),
        ("Avg Speed (Micro-Sector)", "Avg_Speed_MicroSector"),
        ("Speed Std (Micro-Sector)", "Speed_Std_MicroSector"),
        ("Brake Start Distance", "Brake_Start_Distance"),
        ("Brake End Distance", "Brake_End_Distance"),
        ("Braking Zone Length", "Braking_Zone_Length"),
        ("Trail Braking Index", "Trail_Braking_Index"),
        ("Throttle Reapply Distance", "Throttle_Reapply_Distance"),
        ("Avg Throttle After Apex", "Avg_Throttle_After_Apex"),
        ("Full Throttle %", "Full_Throttle_Pct"),
        ("Throttle Commitment Index", "Throttle_Commitment_Index"),
        ("Corner Aggression Score", "Corner_Aggression_Score"),
        ("Smoothness Index", "Smoothness_Index"),
    ]

    rows = []
    for label, col in advanced_metrics:
        if col not in df.columns:
            continue
        v1 = row1[col]
        v2 = row2[col]
        rows.append(
            {
                "Metric": label,
                "Driver 1 value": v1,
                "Driver 2 value": v2,
                "Difference": calculate_metric_difference(v1, v2),
            }
        )
    return pd.DataFrame(rows) if rows else None


def generate_comparison_insight(comparison_df, driver1, driver2):
    insights = []
    metric_map = comparison_df.set_index('Metric')

    entry_v1 = metric_map.loc['Entry Speed', 'Driver 1 value']
    entry_v2 = metric_map.loc['Entry Speed', 'Driver 2 value']
    if entry_v1 > entry_v2:
        insights.append(f"**{driver1}** enters the corner faster than **{driver2}** (higher entry speed).")

    apex_v1 = metric_map.loc['Apex Speed', 'Driver 1 value']
    apex_v2 = metric_map.loc['Apex Speed', 'Driver 2 value']
    if apex_v1 < apex_v2:
        insights.append(f"**{driver1}** slows more at the apex than **{driver2}** (lower apex speed).")

    braking_v1 = metric_map.loc['Braking %', 'Driver 1 value']
    braking_v2 = metric_map.loc['Braking %', 'Driver 2 value']
    if braking_v1 > braking_v2:
        insights.append(
            f"**{driver1}** spends more of the micro-sector braking than **{driver2}** (higher braking %)."
        )

    if not insights:
        return "No notable differences on entry speed, apex speed, or braking % for the selected comparison."
    return '\n\n'.join(insights)


def render_braking_zone_finder(telemetry_data):
    render_section_header(
        "Braking Zone Finder",
        "Inspect loaded telemetry to find braking zones and choose a more meaningful micro-sector window.",
    )
    info_note("Use this to find braking-heavy distance windows instead of guessing the micro-sector manually.")
    render_section_guidance(
        "This finder highlights where speed drops sharply so you can pick a better braking-focused window.",
        "Review the recommended sector first, then inspect the detected brake spans if you want the exact zone boundaries.",
        "A weak sector can still look flat even after detection, so use the recommendation as a starting point instead of a rule.",
    )

    if telemetry_data is None:
        st.info("Run Micro-Sector Analysis to load telemetry before using the braking zone finder.")
        return

    available_drivers = sorted(list(telemetry_data.keys()))
    if not available_drivers:
        st.info("No loaded telemetry is available for braking-zone analysis.")
        return

    selected_driver = st.selectbox(
        "Select driver",
        options=available_drivers,
        key="braking_zone_driver_selector",
    )

    _raw_tel = get_driver_telemetry(telemetry_data, selected_driver)
    if _raw_tel is None or _raw_tel.empty:
        st.warning("Telemetry not available for this driver.")
        return
    selected_telemetry = _raw_tel.copy()
    braking_zones = detect_braking_zones(selected_telemetry)
    if braking_zones.empty:
        st.warning("No braking zones detected for this driver's loaded telemetry.")
        return

    top_zone = braking_zones.iloc[0]
    recommended_start = max(0, round(float(top_zone["Brake_Start_Distance"] - 50), 2))
    recommended_end = round(float(top_zone["Brake_End_Distance"] + 100), 2)
    render_metric_cards(
        [
            ("Detected zones", int(len(braking_zones)), "Number of braking windows found in the loaded telemetry."),
            ("Top zone length", round(float(top_zone["Zone_Length"]), 2), "Length of the strongest braking zone."),
            ("Recommended window", f"{recommended_start} - {recommended_end} m", "Suggested micro-sector start and end based on the strongest braking zone."),
        ]
    )

    shapes = []
    for _, zone in braking_zones.iterrows():
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=float(zone["Brake_Start_Distance"]),
            x1=float(zone["Brake_End_Distance"]),
            y0=0,
            y1=1,
            fillcolor="#f59e0b",
            opacity=0.15,
            line_width=0,
        ))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selected_telemetry["Distance"],
        y=selected_telemetry["Speed"],
        mode="lines",
        line=dict(color="#1d4ed8", width=1.5),
        name="Speed",
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Speed vs Distance for {selected_driver}",
        xaxis_title="Distance (m)",
        yaxis_title="Speed",
        shapes=shapes,
    )
    apply_chart_animation(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The shaded regions mark braking windows where speed drops quickly and the segment becomes informative.")

    with st.expander("View detected braking zones"):
        display_braking_zones = braking_zones.copy()
        display_braking_zones["Brake_Start_Distance"] = display_braking_zones["Brake_Start_Distance"].round(2)
        display_braking_zones["Brake_End_Distance"] = display_braking_zones["Brake_End_Distance"].round(2)
        display_braking_zones["Zone_Length"] = display_braking_zones["Zone_Length"].round(2)
        display_braking_zones["Entry_Speed"] = display_braking_zones["Entry_Speed"].round(2)
        display_braking_zones["Minimum_Speed"] = display_braking_zones["Minimum_Speed"].round(2)
        display_braking_zones["Speed_Loss"] = display_braking_zones["Speed_Loss"].round(2)
        st.dataframe(display_braking_zones, use_container_width=True, hide_index=True)

    st.caption(
        "Select the top braking zone by speed loss, then copy these start/end values into the sidebar and rerun Micro-Sector Analysis."
    )


def render_section_guidance(what_it_tells_you, how_to_read, watch_out_for):
    st.markdown(
        f"""
        <div class="f1-callout">
            <div class="f1-callout__eyebrow">How to read this section</div>
            <div class="f1-callout__body">
                <p><strong>What this section tells you:</strong> {what_it_tells_you}</p>
                <p><strong>How to read it:</strong> {how_to_read}</p>
                <p><strong>What to watch for:</strong> {watch_out_for}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prettify_label(label):
    mapping = {
        "Corner_Aggression_Score": "Aggression score",
        "Apex_Speed": "Apex speed",
        "Braking_Pct": "Braking %",
        "Throttle_Pct": "Throttle %",
        "Throttle_Commitment_Index": "Throttle commitment",
        "Smoothness_Index": "Smoothness",
        "Speed_Loss": "Speed loss",
        "Entry_Speed": "Entry speed",
        "Exit_Speed": "Exit speed",
        "Driving_Profile": "Driving profile",
        "Heavy Trail-Braker": "Heavy trail-braker",
        "High Apex Speed": "High apex speed",
        "Early Accelerator": "Early accelerator",
    }

    if label in mapping:
        return mapping[label]

    if isinstance(label, str):
        cleaned = label.replace("_", " ").strip()
        return " ".join(word.capitalize() for word in cleaned.split())

    return str(label)


def render_metric_card(label, value, caption=None):
    display_label = prettify_label(label)
    display_value = str(value)
    value_class = "metric-card__value"
    if isinstance(value, str) and len(value) > 15:
        value_class = "metric-card__value metric-card__value--compact"

    metric_html = f"""
    <div class="metric-card">
        <div class="metric-card__label">{display_label}</div>
        <div class="{value_class}">{display_value}</div>
        {f'<div class="metric-card__caption">{caption}</div>' if caption else ''}
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)


def render_metric_cards(metrics, per_row=4):
    for start in range(0, len(metrics), per_row):
        chunk = metrics[start:start + per_row]
        columns = st.columns(len(chunk))
        for column, (label, value, caption) in zip(columns, chunk):
            with column:
                render_metric_card(label, value, caption)


def render_section_header(title, description):
    st.markdown(
        f"""
        <div class="f1-section-header">
            <div class="f1-section-header__eyebrow">F1 analytics</div>
            <h2 class="f1-section-header__title">{title}</h2>
            <div class="f1-section-header__accent"></div>
            <p class="f1-section-header__description">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(df, analysis_config, skipped_drivers, last_load_time_sec):
    render_section_header(
        "Overview",
        "Executive summary of the active micro-sector analysis and clustering setup.",
    )
    info_note("This page summarizes the selected race, session, drivers, and the active sector setup. Use it to check whether the chosen windows contain enough braking and speed variation for meaningful analysis.")
    render_section_guidance(
        "This overview shows whether the selected sector produced coherent telemetry, how many samples are loaded, and what driving profile dominates the current run.",
        "Start with the insight cards, then inspect the detailed tables only if needed.",
        "Weak sectors can flatten braking and apex differences, so use the warning and the deeper expanders when the metrics look noisy.",
    )

    session_display_map = dict(zip(SESSION_VALUES, SESSION_LABELS))
    st.markdown("**Configuration summary**")
    sector_mode = analysis_config.get("sector_mode", "Single sector")
    sector_definitions = analysis_config.get("sector_definitions") or []
    sector_count = len(sector_definitions) if sector_definitions else (1 if sector_mode == "Single sector" else 0)
    config_summary = [
        ("Year", analysis_config["year"], "Race year used for telemetry selection."),
        ("Grand Prix", analysis_config["grand_prix"], "Selected event and circuit."),
        ("Session", session_display_map.get(analysis_config["session_type"], analysis_config["session_type"]), "Session type used for telemetry loading."),
        ("Sector mode", sector_mode, "Single-sector or multi-sector extraction mode."),
        ("Number of sectors", sector_count, "How many distance windows were extracted."),
        ("Drivers selected", len(analysis_config["driver_codes"]), "Number of drivers included in the run."),
    ]
    if sector_mode == "Single sector":
        config_summary.append(
            ("Distance window", f"{analysis_config['start_distance']} - {analysis_config['end_distance']} m", "Micro-sector limits applied to the telemetry window.")
        )
    else:
        sector_labels = ", ".join(
            [sector.get("Sector_Name", "Sector") for sector in sector_definitions]
        )
        config_summary.append(("Active sectors", sector_labels, "Distance windows included in the run."))
    if last_load_time_sec is not None:
        config_summary.append(
            ("Load time", f"{last_load_time_sec:.2f}s", "Time spent fetching and preparing the active analysis.")
        )
    for label, value, help_text in config_summary:
        st.markdown(f"**{label}:** {value}")

    avg_speed_loss = float(pd.to_numeric(df["Speed_Loss"], errors="coerce").mean())
    avg_braking_pct = float(pd.to_numeric(df["Braking_Pct"], errors="coerce").mean())
    dominant_profile = "Unknown"
    if "Driving_Profile" in df.columns:
        non_null_profiles = df["Driving_Profile"].dropna()
        if not non_null_profiles.empty:
            dominant_profile = str(non_null_profiles.mode().iloc[0])

    sector_quality_warning = avg_speed_loss < 5 and avg_braking_pct < 5

    driver_sector_samples = int(len(df))
    unique_sectors = int(df["Sector_Name"].nunique()) if "Sector_Name" in df.columns else 1
    render_metric_cards(
        [
            ("Driver-sector samples", driver_sector_samples, "Total driver-sector rows in the current analysis."),
            ("Unique sectors", unique_sectors, "Number of active distance windows included in the run."),
            ("Drivers", int(df["Driver"].nunique()), "Unique driver codes loaded into the analysis."),
            ("Avg speed loss", round(avg_speed_loss, 2), "Average speed drop across the selected micro-sector."),
            ("Avg braking %", round(avg_braking_pct, 2), "Average braking share inside the selected window."),
            ("Dominant profile", dominant_profile, "Most common driving-style label in the current run."),
        ]
    )

    if sector_quality_warning:
        st.warning(
            "This micro-sector looks weak for style separation. Braking and speed-loss variation are low, so interpret the summary cautiously."
        )

    st.markdown("**Detailed diagnostics**")
    diagnostic_rows = [
        {"Metric": "Samples", "Value": len(df)},
        {"Metric": "Unique drivers", "Value": int(df["Driver"].nunique())},
        {"Metric": "Average entry speed", "Value": round(float(pd.to_numeric(df["Entry_Speed"], errors="coerce").mean()), 2)},
        {"Metric": "Average apex speed", "Value": round(float(pd.to_numeric(df["Apex_Speed"], errors="coerce").mean()), 2)},
        {"Metric": "Average speed loss", "Value": round(avg_speed_loss, 2)},
        {"Metric": "Average braking %", "Value": round(avg_braking_pct, 2)},
    ]
    with st.expander("View detailed diagnostics"):
        st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)

    with st.expander("View engineered feature dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    ml_feature_matrix = st.session_state.get("X_imputed_df")
    with st.expander("View ML feature matrix"):
        if ml_feature_matrix is not None:
            st.dataframe(ml_feature_matrix, use_container_width=True, hide_index=True)
        else:
            st.info("Run an analysis to populate the ML feature matrix.")

    with st.expander("View skipped drivers"):
        if skipped_drivers:
            st.dataframe(pd.DataFrame(skipped_drivers), use_container_width=True, hide_index=True)
        else:
            st.info("No drivers were skipped in the active run.")

    st.caption("Use the expanders below to inspect the raw feature tables only when you need to drill into the loaded data.")


def render_telemetry_section(df, telemetry_data):
    render_section_header(
        "Telemetry Trace Analysis",
        "Compare Speed, Throttle, and Brake traces across selected drivers or sectors.",
    )
    info_note(
        "Use this page to inspect raw telemetry. In multi-sector mode, choose whether you want to compare multiple drivers in the same sector or compare multiple sectors for the same driver."
    )
    st.caption(
        "Repeated driver names usually mean the same driver has samples in multiple sectors. Sector labels are shown to make each trace clear."
    )

    if telemetry_data is None:
        st.info("Run Micro-Sector Analysis to load telemetry before using Telemetry Trace Analysis.")
        return

    plot_df = df.copy()
    analysis_config = st.session_state.get("analysis_config") or {}

    if "Sector_Name" not in plot_df.columns:
        plot_df["Sector_Name"] = "Selected sector"
    if "Sector_Start" not in plot_df.columns:
        plot_df["Sector_Start"] = analysis_config.get("start_distance", np.nan)
    if "Sector_End" not in plot_df.columns:
        plot_df["Sector_End"] = analysis_config.get("end_distance", np.nan)

    def build_sector_display_label(row):
        sector_name = row.get("Sector_Name", "Selected sector")
        start = row.get("Sector_Start", np.nan)
        end = row.get("Sector_End", np.nan)

        if pd.notna(start) and pd.notna(end):
            return f"{sector_name} ({int(start)}–{int(end)} m)"
        return str(sector_name)

    plot_df["Sector_Display_Label"] = plot_df.apply(build_sector_display_label, axis=1)
    plot_df["Sample_Label"] = plot_df.apply(
        lambda row: f"{row['Driver']} — {row['Sector_Display_Label']}",
        axis=1,
    )

    telemetry_comparison_mode = st.radio(
        "Telemetry comparison mode",
        options=[
            "Compare drivers in one sector",
            "Compare sectors for one driver",
        ],
        index=0,
        key="telemetry_comparison_mode",
    )

    skipped_labels = []
    trace_specs = []

    if telemetry_comparison_mode == "Compare drivers in one sector":
        sector_options = sorted(plot_df["Sector_Display_Label"].drop_duplicates().tolist())
        if not sector_options:
            st.warning("No sector labels are available for telemetry plotting.")
            return

        selected_sector_label = st.selectbox(
            "Select sector",
            options=sector_options,
            key="telemetry_selected_sector",
        )
        selected_sector_rows = plot_df[plot_df["Sector_Display_Label"] == selected_sector_label].copy()
        driver_options = sorted(
            selected_sector_rows["Driver"].drop_duplicates().astype(str).tolist()
        )
        selected_drivers = st.multiselect(
            "Select drivers",
            options=driver_options,
            default=driver_options[:3],
            key="telemetry_selected_drivers",
        )

        if not selected_drivers:
            st.warning("Please select at least one driver to view telemetry traces.")
            return

        for driver in selected_drivers:
            driver_rows = selected_sector_rows[selected_sector_rows["Driver"] == driver]
            if driver_rows.empty:
                continue
            sample_row = driver_rows.iloc[0]
            telemetry = get_driver_telemetry(telemetry_data, str(driver))
            trace_specs.append(
                {
                    "label": str(driver),
                    "telemetry": telemetry,
                    "sector_start": sample_row["Sector_Start"],
                    "sector_end": sample_row["Sector_End"],
                }
            )
        st.caption("This mode compares different drivers over the same selected sector.")
    else:
        driver_options = sorted(plot_df["Driver"].drop_duplicates().astype(str).tolist())
        selected_driver = st.selectbox(
            "Select driver",
            options=driver_options,
            key="telemetry_selected_driver_for_sector_compare",
        )
        driver_rows = plot_df[plot_df["Driver"] == selected_driver].copy()
        sector_options = sorted(driver_rows["Sector_Display_Label"].drop_duplicates().tolist())
        if not sector_options:
            st.warning("No sector labels are available for telemetry plotting.")
            return

        selected_sectors = st.multiselect(
            "Select sectors",
            options=sector_options,
            default=sector_options[:3],
            key="telemetry_selected_sectors",
        )

        if not selected_sectors:
            st.warning("Please select at least one sector to view telemetry traces.")
            return

        for sector_label in selected_sectors:
            sector_rows = driver_rows[driver_rows["Sector_Display_Label"] == sector_label]
            if sector_rows.empty:
                continue
            sample_row = sector_rows.iloc[0]
            telemetry = get_driver_telemetry(telemetry_data, str(selected_driver))
            trace_specs.append(
                {
                    "label": str(sector_label),
                    "telemetry": telemetry,
                    "sector_start": sample_row["Sector_Start"],
                    "sector_end": sample_row["Sector_End"],
                }
            )
        st.caption("This mode compares how the same driver behaves across different selected sectors.")

    for column, chart_title, ylabel in [
        ("Speed", "Speed vs Distance (Micro-Sector)", "Speed (km/h)"),
        ("Throttle", "Throttle vs Distance (Micro-Sector)", "Throttle (%)"),
        ("Brake", "Brake vs Distance (Micro-Sector)", "Brake"),
    ]:
        current_fig, skipped = plot_telemetry_traces(
            trace_specs,
            column,
            chart_title,
            ylabel,
        )
        skipped_labels.extend(skipped)

        if current_fig is False:
            st.warning(
                "No telemetry points found for the selected driver-sector combination. Try widening the sector range or choose another sector."
            )
            continue

        if len(skipped) > 0 and len(skipped) != len(trace_specs):
            st.warning("Some selected traces had no telemetry points in this sector and were skipped.")

        apply_chart_animation(current_fig)
        st.plotly_chart(current_fig, use_container_width=True)


def render_driver_comparison(df):
    render_section_header(
        "Driver Comparison Mode",
        "Compare two drivers side by side using micro-sector feature metrics.",
    )
    info_note("Compare two drivers within the selected micro-sector. Results depend only on the loaded race, session, and distance window.")
    comparison_df = df.copy()
    if "Sector_Name" in comparison_df.columns and comparison_df["Sector_Name"].nunique() > 1:
        sector_options = comparison_df["Sector_Name"].drop_duplicates().tolist()
        selected_sector = st.selectbox(
            "Sector filter",
            options=sector_options,
            key="comparison_sector_filter",
        )
        comparison_df = comparison_df[comparison_df["Sector_Name"] == selected_sector].copy()
        st.caption(f"Showing comparison for sector: {selected_sector}")

    available_drivers = comparison_df['Driver'].tolist()
    if len(available_drivers) < 2:
        st.warning("At least two drivers are required for side-by-side comparison.")
        return

    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox(
            "Select Driver 1",
            options=available_drivers,
            index=0,
            key="comparison_driver_1",
        )
    with col2:
        driver2 = st.selectbox(
            "Select Driver 2",
            options=available_drivers,
            index=1,
            key="comparison_driver_2",
        )

    comparison_df = build_driver_comparison(comparison_df, driver1, driver2)
    if comparison_df is not None:
        display_comparison_df = comparison_df.copy()
        display_comparison_df['Driver 1 value'] = display_comparison_df['Driver 1 value'].apply(format_metric_value)
        display_comparison_df['Driver 2 value'] = display_comparison_df['Driver 2 value'].apply(format_metric_value)
        display_comparison_df['Difference'] = display_comparison_df['Difference'].apply(format_metric_value)
        st.dataframe(display_comparison_df, use_container_width=True, hide_index=True)
        st.markdown("**Insight**")
        st.markdown(generate_comparison_insight(comparison_df, driver1, driver2))

    st.caption("Not detected usually means the selected window did not contain braking or throttle reapplication for that driver.")
    with st.expander("Advanced telemetry metrics"):
        st.markdown(
            "Some braking metrics may show 'Not detected' if the selected micro-sector does not contain active braking telemetry."
        )
        st.caption("Aggression score is a composite index from normalized speed loss, braking involvement, and throttle commitment. It is not an official F1 metric.")
        adv_df = build_driver_comparison_advanced(df, driver1, driver2)
        if adv_df is None or adv_df.empty:
            st.info("Advanced metrics are not available for this configuration.")
        else:
            display_adv_df = adv_df.copy()
            display_adv_df['Driver 1 value'] = display_adv_df['Driver 1 value'].apply(format_metric_value)
            display_adv_df['Driver 2 value'] = display_adv_df['Driver 2 value'].apply(format_metric_value)
            display_adv_df['Difference'] = display_adv_df['Difference'].apply(format_metric_value)
            st.dataframe(display_adv_df, use_container_width=True, hide_index=True)


def render_cluster_visualization(df):
    render_section_header(
        "Cluster Visualization",
        "Visualize driver groups by apex speed and braking usage.",
    )
    info_note("Clustering groups telemetry samples with similar behavior. These labels are exploratory and depend on the selected micro-sector.")
    required_columns = {"Apex_Speed", "Braking_Pct", "Driving_Profile"}
    missing_cluster_columns = sorted(required_columns - set(df.columns))
    if missing_cluster_columns:
        st.warning(
            "Cluster visualization is unavailable because required columns are missing: "
            + ", ".join(missing_cluster_columns)
        )
        return

    has_sectors = 'Sector_Name' in df.columns
    rng = np.random.default_rng(42)
    scatter_df = df.copy()
    scatter_df['Apex_Speed_jittered'] = (
        scatter_df['Apex_Speed'].astype(float) + rng.uniform(-0.3, 0.3, len(df))
    )

    profiles = sorted(scatter_df['Driving_Profile'].unique())
    sectors = sorted(scatter_df['Sector_Name'].unique()) if has_sectors else ['All']

    color_seq = px.colors.qualitative.Plotly
    profile_colors = {p: color_seq[i % len(color_seq)] for i, p in enumerate(profiles)}
    symbol_seq = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x']
    sector_symbols = {s: symbol_seq[i % len(symbol_seq)] for i, s in enumerate(sectors)}

    fig = go.Figure()

    for profile in profiles:
        for sector in sectors:
            if has_sectors:
                mask = (scatter_df['Driving_Profile'] == profile) & (scatter_df['Sector_Name'] == sector)
            else:
                mask = scatter_df['Driving_Profile'] == profile
            sub = scatter_df[mask]
            if sub.empty:
                continue
            sector_vals = sub['Sector_Name'].values if has_sectors else np.array(['All'] * len(sub))
            customdata = np.stack([
                sub['Driver'].values,
                sub['Driving_Profile'].values,
                sector_vals,
                sub['Apex_Speed'].values.astype(float),
                sub['Braking_Pct'].values.astype(float),
            ], axis=1)
            fig.add_trace(go.Scatter(
                x=sub['Apex_Speed_jittered'],
                y=sub['Braking_Pct'],
                mode='markers',
                showlegend=False,
                marker=dict(color=profile_colors[profile], symbol=sector_symbols[sector], size=10),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Profile: %{customdata[1]}<br>"
                    "Sector: %{customdata[2]}<br>"
                    "Apex Speed: %{customdata[3]:.1f} km/h<br>"
                    "Braking: %{customdata[4]:.1f}%<extra></extra>"
                ),
            ))

    for i, profile in enumerate(profiles):
        extra = dict(legendgrouptitle=dict(text='Driving Profile')) if i == 0 else {}
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name=profile,
            legendgroup='profile',
            marker=dict(color=profile_colors[profile], symbol='circle', size=10),
            showlegend=True,
            **extra,
        ))

    if has_sectors:
        for i, sector in enumerate(sectors):
            extra = dict(legendgrouptitle=dict(text='Sector')) if i == 0 else {}
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name=sector,
                legendgroup='sector',
                marker=dict(color='rgba(150,150,150,0.8)', symbol=sector_symbols[sector], size=10),
                showlegend=True,
                **extra,
            ))

    fig.update_layout(
        title="Driver Clusters by Apex Speed and Braking Percentage",
        xaxis_title='Apex Speed (km/h)',
        yaxis_title='Braking Usage in Micro-Sector (%)',
        legend=dict(groupclick='toggleitem'),
    )
    apply_chart_animation(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher apex speed means the driver carries more speed at the slowest point of the window. Higher braking % means braking is active for more of the window.")

    cluster_table = (
        df[['Driver', 'Apex_Speed', 'Braking_Pct', 'Driving_Profile']]
        .sort_values('Driving_Profile')
        .reset_index(drop=True)
    )
    st.dataframe(cluster_table, use_container_width=True, hide_index=True)

    # PCA biplot
    st.markdown("---")
    st.markdown("**PCA Biplot**")
    X_scaled_pca = st.session_state.get("X_scaled")
    used_feature_cols_pca = st.session_state.get("used_feature_columns") or []

    if X_scaled_pca is None or len(used_feature_cols_pca) < 2:
        st.info("PCA biplot is unavailable — run the analysis first.")
    elif X_scaled_pca.shape[0] != len(df):
        st.info("PCA biplot is unavailable — sample count mismatch between feature matrix and current view.")
    else:
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled_pca)
        loadings = pca.components_.T  # shape (n_features, 2)
        var_explained = pca.explained_variance_ratio_
        scale = 3.0

        pca_plot_df = pd.DataFrame({
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "Driver": df["Driver"].values,
            "Driving_Profile": df["Driving_Profile"].values,
        })

        pca_fig = px.scatter(
            pca_plot_df,
            x="PC1",
            y="PC2",
            color="Driving_Profile",
            hover_data=["Driver"],
            text="Driver",
            title=(
                f"PCA Biplot — PC1 ({var_explained[0]:.1%} variance) "
                f"vs PC2 ({var_explained[1]:.1%} variance)"
            ),
            labels={
                "PC1": f"PC1 ({var_explained[0]:.1%})",
                "PC2": f"PC2 ({var_explained[1]:.1%})",
                "Driving_Profile": "Driving Profile",
            },
        )
        pca_fig.update_traces(textposition="top center", marker_size=9)

        shapes = []
        annotations = []
        for i, feature in enumerate(used_feature_cols_pca):
            lx = float(loadings[i, 0]) * scale
            ly = float(loadings[i, 1]) * scale
            shapes.append(dict(
                type="line",
                xref="x",
                yref="y",
                x0=0,
                y0=0,
                x1=lx,
                y1=ly,
                line=dict(color="rgba(220,50,50,0.65)", width=1.5),
            ))
            annotations.append(dict(
                x=lx,
                y=ly,
                xref="x",
                yref="y",
                text=prettify_label(feature),
                showarrow=False,
                font=dict(size=11, color="rgba(220,50,50,0.9)"),
                bgcolor="white",
                opacity=0.7,
            ))

        pca_fig.update_layout(shapes=shapes, annotations=annotations)
        apply_chart_animation(pca_fig)
        st.plotly_chart(pca_fig, use_container_width=True)
        st.caption("Each point represents one driver-sector sample. Drivers with multiple sectors appear more than once.")

        with st.expander("What is this?"):
            st.markdown(
                "Each arrow shows how strongly a feature pulls clusters apart. "
                "Arrows pointing in the same direction mean those features are correlated. "
                "A driver plotted near an arrow tip scores high on that feature."
            )


def build_top5_lap_features(top5_data, sector_definitions, year, grand_prix, session_type):
    """Build one row per driver-lap-sector from top-5-laps telemetry data."""
    rows = []
    for driver_code, laps_list in top5_data.items():
        for lap_number, lap_time, tel in laps_list:
            for sector in sector_definitions:
                feat = extract_turn_features(
                    driver_code,
                    tel,
                    start_distance=sector["Sector_Start"],
                    end_distance=sector["Sector_End"],
                    sector_name=sector["Sector_Name"],
                )
                if feat is not None:
                    feat["Lap_Number"] = lap_number
                    feat["LapTime"] = round(lap_time, 3)
                    feat["Year"] = year
                    feat["Grand_Prix"] = grand_prix
                    feat["Session_Type"] = session_type
                    rows.append(feat)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_consistency_metrics(raw_df):
    """Aggregate per-lap-sector rows into per-driver-sector consistency scores."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["Driver", "Sector_Name"] if c in raw_df.columns]
    if not group_cols:
        return pd.DataFrame()

    available_metrics = [m for m in CONSISTENCY_METRICS if m in raw_df.columns]

    results = []
    for group_keys, group_df in raw_df.groupby(group_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(group_cols, group_keys))
        row["Sample_Count"] = len(group_df)
        for metric in available_metrics:
            vals = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            row[f"{metric}_Mean"] = round(float(vals.mean()), 3) if len(vals) > 0 else np.nan
            row[f"{metric}_Std"] = round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else np.nan
        results.append(row)

    if not results:
        return pd.DataFrame()

    summary_df = pd.DataFrame(results)

    norm_stds = pd.DataFrame(index=summary_df.index)
    for metric in available_metrics:
        std_col = f"{metric}_Std"
        if std_col in summary_df.columns:
            max_std = summary_df[std_col].dropna().max()
            if pd.isna(max_std) or max_std < 1e-9:
                norm_stds[metric] = 0.0
            else:
                norm_stds[metric] = (summary_df[std_col].fillna(0.0) / max_std).clip(0.0, 1.0)

    if not norm_stds.empty and len(norm_stds.columns) > 0:
        avg_norm_std = norm_stds.mean(axis=1)
        summary_df["Consistency_Score"] = (100.0 * (1.0 - avg_norm_std)).clip(0, 100).round(1)
    else:
        summary_df["Consistency_Score"] = np.nan

    summary_df.loc[summary_df["Sample_Count"] < 2, "Consistency_Score"] = np.nan

    return (
        summary_df
        .sort_values("Consistency_Score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def render_consistency_analysis():
    render_section_header(
        "Consistency Analysis",
        "How repeatable is each driver's micro-sector technique across their top 5 laps?",
    )
    st.caption(
        "Consistency measures how tightly a driver's braking point, apex speed, and throttle commitment "
        "cluster across repeated laps — lower lap-to-lap variation means a higher consistency score."
    )
    st.info(
        "Consistency metrics require at least 2 valid laps per driver. In Qualifying sessions, "
        "some drivers may not have enough laps — Race or Practice sessions are recommended for "
        "consistency analysis."
    )

    summary_df = st.session_state.get("consistency_summary_df")
    raw_df = st.session_state.get("consistency_raw_df")

    if summary_df is None or summary_df.empty:
        st.warning(
            "No consistency data is available. "
            "Select 'Top 5 laps consistency' lap mode and run the analysis first."
        )
        return

    if summary_df["Consistency_Score"].isna().all():
        st.warning(
            "No drivers had sufficient laps for consistency analysis. "
            "Try a Race or Practice session."
        )
        return

    # --- Driver consistency ranking table ---
    st.markdown("**Driver Consistency Ranking**")
    display_cols = ["Driver"]
    if "Sector_Name" in summary_df.columns and summary_df["Sector_Name"].nunique() > 1:
        display_cols.append("Sector_Name")
    display_cols += ["Sample_Count", "Consistency_Score"]
    for metric in CONSISTENCY_METRICS:
        for suffix in ["_Mean", "_Std"]:
            col = metric + suffix
            if col in summary_df.columns:
                display_cols.append(col)
    display_cols = list(dict.fromkeys(c for c in display_cols if c in summary_df.columns))

    display_ranking = summary_df[display_cols].copy()
    for _col in display_ranking.columns:
        if _col.endswith("_Mean") or _col == "Consistency_Score":
            display_ranking[_col] = pd.to_numeric(display_ranking[_col], errors="coerce").round(1)
        elif _col.endswith("_Std"):
            display_ranking[_col] = pd.to_numeric(display_ranking[_col], errors="coerce").round(2)
    if "Consistency_Score" in display_ranking.columns:
        display_ranking["Consistency_Score"] = display_ranking["Consistency_Score"].apply(
            lambda v: "Unavailable (< 2 laps)" if pd.isna(v) else v
        )
    st.dataframe(display_ranking, use_container_width=True, hide_index=True)

    skipped_cons_drivers = st.session_state.get("consistency_skipped_drivers") or []
    if skipped_cons_drivers:
        skipped_msgs = ", ".join(
            f"{d['Driver']} ({d.get('Reason', 'skipped')})"
            for d in skipped_cons_drivers
        )
        st.warning(f"Skipped drivers (insufficient laps): {skipped_msgs}")

    # --- Bar chart ---
    chart_df = summary_df.dropna(subset=["Consistency_Score"]).copy()
    if chart_df.empty:
        st.warning("No drivers have enough laps to compute a consistency score.")
    else:
        chart_df = chart_df.sort_values("Consistency_Score", ascending=True).reset_index(drop=True)
        x_col = "Driver"
        if "Sector_Name" in chart_df.columns:
            if chart_df["Sector_Name"].nunique() > 1:
                chart_df["Driver_Sector"] = (
                    chart_df["Driver"].astype(str) + " | " + chart_df["Sector_Name"].astype(str)
                )
                x_col = "Driver_Sector"
            else:
                chart_df = chart_df[
                    chart_df["Sector_Name"] == chart_df["Sector_Name"].iloc[0]
                ].copy()
        fig = px.bar(
            chart_df,
            x="Consistency_Score",
            y=x_col,
            orientation="h",
            text=chart_df["Consistency_Score"].round(1),
            labels={"Consistency_Score": "Consistency Score (0–100)", x_col: "Driver"},
            title="Driver Consistency Score — higher means more repeatable technique",
            color="Consistency_Score",
            color_continuous_scale=[[0.0, "#7f0000"], [0.5, "#e10600"], [1.0, "#22c55e"]],
            range_color=[0, 100],
        )
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        apply_chart_animation(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "A score near 100 means near-zero lap-to-lap variation across all tracked metrics. "
            "Scores above 80 indicate strong technique repeatability."
        )

    # --- Selected driver detail ---
    st.markdown("**Driver Detail View**")
    available_cons_drivers = summary_df["Driver"].unique().tolist()
    if not available_cons_drivers:
        return
    selected_cons_driver = st.selectbox(
        "Select driver for detail",
        options=available_cons_drivers,
        key="consistency_detail_driver",
    )

    driver_summary_rows = summary_df[summary_df["Driver"] == selected_cons_driver]
    driver_raw = (
        raw_df[raw_df["Driver"] == selected_cons_driver].copy()
        if (raw_df is not None and not raw_df.empty)
        else pd.DataFrame()
    )

    for _, driver_row in driver_summary_rows.iterrows():
        sector_label = (
            str(driver_row["Sector_Name"]) if "Sector_Name" in driver_row.index else "All sectors"
        )
        sample_count = int(driver_row.get("Sample_Count", 0))
        cons_score = driver_row.get("Consistency_Score", np.nan)

        st.markdown(
            f"**{selected_cons_driver} — {sector_label}** "
            f"({sample_count} lap{'s' if sample_count != 1 else ''})"
        )

        if sample_count < 2:
            st.warning(
                f"Fewer than 2 valid laps were found for {selected_cons_driver} in "
                f"sector '{sector_label}'. Consistency score is unavailable."
            )
            continue

        score_display = f"{cons_score:.1f} / 100" if pd.notna(cons_score) else "Unavailable"
        st.metric("Consistency Score", score_display)

        available_lap_metrics = [m for m in CONSISTENCY_METRICS if m in driver_raw.columns]
        if not driver_raw.empty:
            if "Sector_Name" in driver_raw.columns:
                sector_raw = driver_raw[driver_raw["Sector_Name"] == sector_label].copy()
            else:
                sector_raw = driver_raw.copy()
            if not sector_raw.empty:
                show_cols = [
                    c for c in ["Lap_Number", "LapTime"] + available_lap_metrics
                    if c in sector_raw.columns
                ]
                if show_cols:
                    st.markdown("Top 5 lap feature values:")
                    sort_col = "Lap_Number" if "Lap_Number" in sector_raw.columns else show_cols[0]
                    st.dataframe(
                        sector_raw[show_cols].sort_values(sort_col).round(3),
                        use_container_width=True,
                        hide_index=True,
                    )

        std_values = {
            metric: float(driver_row[f"{metric}_Std"])
            for metric in CONSISTENCY_METRICS
            if f"{metric}_Std" in driver_row.index and pd.notna(driver_row[f"{metric}_Std"])
        }
        if std_values:
            most_variable_metric = max(std_values, key=std_values.get)
            st.markdown(
                f"Most variable metric: **{prettify_label(most_variable_metric)}** "
                f"(std = {std_values[most_variable_metric]:.3f})"
            )

    # --- Auto-insight ---
    st.markdown("**Auto Insight**")
    valid_scores = summary_df.dropna(subset=["Consistency_Score"])
    if valid_scores.empty:
        st.info("No valid consistency scores are available for this session.")
        return

    most_row = valid_scores.loc[valid_scores["Consistency_Score"].idxmax()]
    least_row = valid_scores.loc[valid_scores["Consistency_Score"].idxmin()]
    most_driver = str(most_row["Driver"])
    most_score = round(float(most_row["Consistency_Score"]), 1)
    least_driver = str(least_row["Driver"])
    least_score = round(float(least_row["Consistency_Score"]), 1)

    # Feature with highest average std across all drivers in the first sector
    if "Sector_Name" in summary_df.columns:
        first_sector = summary_df["Sector_Name"].iloc[0]
        sector_for_var = summary_df[summary_df["Sector_Name"] == first_sector]
    else:
        sector_for_var = summary_df
    std_cols_var = [
        f"{m}_Std" for m in CONSISTENCY_METRICS if f"{m}_Std" in sector_for_var.columns
    ]
    most_variable_feature = "N/A"
    if std_cols_var:
        avg_stds = sector_for_var[std_cols_var].mean()
        if avg_stds.notna().any():
            most_variable_feature = prettify_label(avg_stds.idxmax().replace("_Std", ""))

    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.metric("Most Consistent", most_driver, f"{most_score}/100", delta_color="off")
    with insight_cols[1]:
        st.metric("Least Consistent", least_driver, f"{least_score}/100", delta_color="off")
    with insight_cols[2]:
        st.metric("Most Variable Feature", most_variable_feature)


load_css()

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("**F1 Telemetry Analytics**")
st.sidebar.caption("Micro-sector driving style clustering")
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-label">Data Configuration</div>', unsafe_allow_html=True)
selected_year = st.sidebar.selectbox(
    "Season Year",
    options=[2023, 2024, 2025],
    index=[2023, 2024, 2025].index(DEFAULT_YEAR),
    key="selected_year_config",
)
selected_gp = st.sidebar.selectbox(
    "Grand Prix",
    options=GP_OPTIONS,
    index=GP_OPTIONS.index(DEFAULT_GP),
    key="selected_gp_config",
)
selected_session_label = st.sidebar.selectbox(
    "Session",
    options=SESSION_LABELS,
    index=SESSION_LABELS.index("Qualifying"),
    key="selected_session_config",
)
selected_session = SESSION_VALUES[SESSION_LABELS.index(selected_session_label)]
selected_drivers_config = st.sidebar.multiselect(
    "Driver Selection",
    options=DEFAULT_DRIVERS,
    default=DEFAULT_DRIVERS[:10],
    key="selected_drivers_config",
)
selected_lap_mode = st.sidebar.radio(
    "Lap mode",
    options=["Fastest lap only", "Top 5 laps consistency"],
    index=["Fastest lap only", "Top 5 laps consistency"].index(
        st.session_state.get("lap_mode", "Fastest lap only")
    ),
    key="selected_lap_mode_config",
)
st.session_state["lap_mode"] = selected_lap_mode
selected_sector_mode = st.sidebar.selectbox(
    "Sector mode",
    options=["Single sector", "Multi-sector"],
    index=["Single sector", "Multi-sector"].index(
        st.session_state["analysis_config"].get("sector_mode", "Single sector")
    ),
    key="selected_sector_mode_config",
)

sector_definitions = []
selected_start_distance = DEFAULT_START_DISTANCE
selected_end_distance = DEFAULT_END_DISTANCE

if selected_sector_mode == "Single sector":
    selected_start_distance = st.sidebar.number_input(
        "Micro-sector start distance (m)",
        min_value=0,
        max_value=7000,
        value=DEFAULT_START_DISTANCE,
        step=50,
        key="selected_start_distance_config",
    )
    selected_end_distance = st.sidebar.number_input(
        "Micro-sector end distance (m)",
        min_value=0,
        max_value=7000,
        value=DEFAULT_END_DISTANCE,
        step=50,
        key="selected_end_distance_config",
    )
    sector_definitions = [
        {
            "Sector_Name": "Selected sector",
            "Sector_Start": int(selected_start_distance),
            "Sector_End": int(selected_end_distance),
        }
    ]
else:
    st.sidebar.markdown("**Multi-sector windows**")
    sector_1_name = st.sidebar.text_input(
        "Sector 1 name",
        value="Sector A",
        key="sector_1_name",
    )
    sector_1_start = st.sidebar.number_input(
        "Sector 1 start distance (m)",
        min_value=0,
        max_value=7000,
        value=300,
        step=50,
        key="sector_1_start",
    )
    sector_1_end = st.sidebar.number_input(
        "Sector 1 end distance (m)",
        min_value=0,
        max_value=7000,
        value=600,
        step=50,
        key="sector_1_end",
    )

    sector_2_enabled = st.sidebar.checkbox("Enable Sector 2", value=True, key="sector_2_enabled")
    sector_2_name = st.sidebar.text_input(
        "Sector 2 name",
        value="Sector B",
        key="sector_2_name",
    )
    sector_2_start = st.sidebar.number_input(
        "Sector 2 start distance (m)",
        min_value=0,
        max_value=7000,
        value=900,
        step=50,
        key="sector_2_start",
    )
    sector_2_end = st.sidebar.number_input(
        "Sector 2 end distance (m)",
        min_value=0,
        max_value=7000,
        value=1200,
        step=50,
        key="sector_2_end",
    )

    sector_3_enabled = st.sidebar.checkbox("Enable Sector 3", value=True, key="sector_3_enabled")
    sector_3_name = st.sidebar.text_input(
        "Sector 3 name",
        value="Sector C",
        key="sector_3_name",
    )
    sector_3_start = st.sidebar.number_input(
        "Sector 3 start distance (m)",
        min_value=0,
        max_value=7000,
        value=1500,
        step=50,
        key="sector_3_start",
    )
    sector_3_end = st.sidebar.number_input(
        "Sector 3 end distance (m)",
        min_value=0,
        max_value=7000,
        value=1800,
        step=50,
        key="sector_3_end",
    )

    sector_definitions = [
        {
            "Sector_Name": sector_1_name,
            "Sector_Start": int(sector_1_start),
            "Sector_End": int(sector_1_end),
        }
    ]
    if sector_2_enabled:
        sector_definitions.append(
            {
                "Sector_Name": sector_2_name,
                "Sector_Start": int(sector_2_start),
                "Sector_End": int(sector_2_end),
            }
        )
    if sector_3_enabled:
        sector_definitions.append(
            {
                "Sector_Name": sector_3_name,
                "Sector_Start": int(sector_3_start),
                "Sector_End": int(sector_3_end),
            }
        )
    selected_start_distance = sector_definitions[0]["Sector_Start"]
    selected_end_distance = sector_definitions[0]["Sector_End"]

selected_n_clusters = st.sidebar.slider(
    "Number of clusters",
    min_value=2,
    max_value=6,
    value=st.session_state["analysis_config"]["n_clusters"],
    key="n_clusters_config",
)

if st.sidebar.button("Run Micro-Sector Analysis", type="primary", use_container_width=True):
    if not selected_drivers_config:
        st.sidebar.warning("Select at least one driver before running analysis.")
    else:
        sector_validation_errors = []
        if not sector_definitions:
            sector_validation_errors.append("Enable at least one sector.")
        for sector in sector_definitions:
            if sector["Sector_Start"] < 0 or sector["Sector_End"] < 0:
                sector_validation_errors.append("Sector distances cannot be negative.")
                break
            if sector["Sector_End"] <= sector["Sector_Start"]:
                sector_validation_errors.append(
                    f"Sector '{sector['Sector_Name']}' must have end distance greater than start distance."
                )
                break

        if sector_validation_errors:
            st.sidebar.error(sector_validation_errors[0])
        else:
            selected_drivers_tuple = tuple(selected_drivers_config)
            sector_definitions_tuple = tuple(
                (sector["Sector_Name"], sector["Sector_Start"], sector["Sector_End"])
                for sector in sector_definitions
            )
            run_config = {
                "year": selected_year,
                "grand_prix": selected_gp,
                "session_type": selected_session,
                "driver_codes": list(selected_drivers_config),
                "start_distance": int(selected_start_distance),
                "end_distance": int(selected_end_distance),
                "sector_mode": selected_sector_mode,
                "sector_definitions": sector_definitions,
                "n_clusters": int(selected_n_clusters),
            }
            st.session_state["analysis_config"] = run_config

            try:
                with st.spinner("Loading FastF1 session and processing telemetry. First load may take longer."):
                    start_time = time.perf_counter()
                    telemetry_data, skipped_drivers = fetch_raw_telemetry_cached(
                        selected_year,
                        selected_gp,
                        selected_session,
                        selected_drivers_tuple,
                    )

                    try:
                        df = build_features_cached(
                            telemetry_data,
                            sector_definitions_tuple,
                            selected_year,
                            selected_gp,
                            selected_session,
                            selected_sector_mode,
                        )
                    except Exception as exc:
                        logger.warning(f"Cached feature build failed, falling back to direct build: {exc}")
                        df = build_feature_dataset(
                            telemetry_data,
                            sector_definitions=sector_definitions,
                            metadata={
                                "Year": selected_year,
                                "Grand_Prix": selected_gp,
                                "Session_Type": selected_session,
                                "Sector_Mode": selected_sector_mode,
                            },
                        )

                    if "Sample_Label" not in df.columns:
                        df = df.copy()
                        df["Sample_Label"] = df.apply(build_sample_label, axis=1)

                    # Store raw driver-keyed telemetry immediately so it's always
                    # available for telemetry trace plotting regardless of whether
                    # clustering succeeds.
                    st.session_state["telemetry_data"] = telemetry_data

                    sample_count = len(df)
                    if sample_count < 2:
                        st.sidebar.error("Not enough valid samples for clustering.")
                        st.session_state["analysis_ready"] = False
                    else:
                        missing_feature_columns = [
                            column for column in CLUSTER_FEATURE_COLUMNS if column not in df.columns
                        ]
                        if missing_feature_columns:
                            st.sidebar.warning(
                                "Missing optional clustering features: " + ", ".join(missing_feature_columns)
                            )
                        st.session_state["missing_feature_columns"] = missing_feature_columns

                        max_valid_clusters = max(1, sample_count - 1)
                        requested_n_clusters = int(selected_n_clusters)
                        if requested_n_clusters >= sample_count:
                            adjusted_n_clusters = max_valid_clusters
                            st.sidebar.warning(
                                f"Requested {requested_n_clusters} clusters, but only {sample_count} samples are available. "
                                f"Reducing to {adjusted_n_clusters}."
                            )
                            requested_n_clusters = adjusted_n_clusters
                        X_scaled, X_imputed_df, imputer, scaler, used_feature_columns = prepare_ml_features(
                            df,
                            CLUSTER_FEATURE_COLUMNS,
                        )
                        if X_scaled.shape[0] < 2:
                            st.sidebar.error("Not enough valid samples for clustering.")
                            st.session_state["analysis_ready"] = False
                        else:
                            df, kmeans_model = run_kmeans_clustering(
                                df,
                                X_scaled,
                                requested_n_clusters,
                            )
                            elapsed = time.perf_counter() - start_time
                            st.session_state["df"] = df
                            st.session_state["skipped_drivers"] = skipped_drivers
                            st.session_state["analysis_ready"] = True
                            st.session_state["last_load_time_sec"] = elapsed
                            st.session_state["X_scaled"] = X_scaled
                            st.session_state["X_imputed_df"] = X_imputed_df
                            st.session_state["kmeans_model"] = kmeans_model
                            st.session_state["used_feature_columns"] = used_feature_columns
                            st.session_state["analysis_config"]["n_clusters"] = requested_n_clusters
                            if selected_lap_mode == "Top 5 laps consistency":
                                try:
                                    top5_data, top5_skipped = fetch_top5_telemetry_cached(
                                        selected_year,
                                        selected_gp,
                                        selected_session,
                                        selected_drivers_tuple,
                                    )
                                    if top5_data:
                                        consistency_raw_df = build_top5_lap_features(
                                            top5_data,
                                            sector_definitions,
                                            selected_year,
                                            selected_gp,
                                            selected_session,
                                        )
                                        st.session_state["consistency_raw_df"] = consistency_raw_df
                                        st.session_state["consistency_summary_df"] = (
                                            compute_consistency_metrics(consistency_raw_df)
                                        )
                                        st.session_state["consistency_skipped_drivers"] = top5_skipped
                                    else:
                                        st.session_state["consistency_raw_df"] = None
                                        st.session_state["consistency_summary_df"] = None
                                        st.session_state["consistency_skipped_drivers"] = top5_skipped
                                except Exception as _top5_exc:
                                    logger.warning(f"Top-5 consistency fetch failed: {_top5_exc}")
                                    st.session_state["consistency_raw_df"] = None
                                    st.session_state["consistency_summary_df"] = None
                                    st.session_state["consistency_skipped_drivers"] = []
            except ValueError as exc:
                st.sidebar.error(str(exc))
                if st.session_state.get("analysis_ready", False):
                    st.sidebar.info("Previous loaded analysis is still displayed.")
            except Exception as exc:
                logger.exception("Analysis failed")
                st.sidebar.error(f"New analysis failed: {exc}")
                if st.session_state.get("analysis_ready", False):
                    st.sidebar.info("Previous loaded analysis is still displayed.")

            if st.session_state.get("analysis_ready", False):
                st.success("Analysis Complete!")

st.sidebar.caption("Load telemetry once, then explore sections.")
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
st.sidebar.radio(
    "Choose analysis section",
    options=[
        "Overview",
        "Telemetry Trace Analysis",
        "Driver Comparison",
        "Cluster Visualization",
        "Cluster Profiles",
        "Driver Profiles",
        "Braking Zone Finder",
        "Model Evaluation",
        "Sector Comparison",
        "Dataset Export",
        "Consistency Analysis",
    ],
    key="selected_section",
)
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-label">Current Setup</div>', unsafe_allow_html=True)
st.sidebar.caption(f"Year: {selected_year}")
st.sidebar.caption(f"GP: {selected_gp}")
st.sidebar.caption(f"Session: {selected_session_label}")
st.sidebar.caption(f"Sector mode: {selected_sector_mode}")
if selected_sector_mode == "Single sector":
    st.sidebar.caption(f"Micro-sector: {selected_start_distance}-{selected_end_distance} m")
else:
    st.sidebar.caption(f"Active sectors: {len(sector_definitions)}")
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-label">Analysis Status</div>', unsafe_allow_html=True)
if st.session_state["analysis_ready"]:
    st.sidebar.markdown(
        '<span class="status-badge status-loaded">Analysis loaded</span>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<span class="status-badge status-not-loaded">Analysis not loaded</span>',
        unsafe_allow_html=True,
    )
with st.sidebar.expander("Cache controls"):
    if st.button("Clear processed cache", key="clear_processed_cache_btn"):
        st.cache_data.clear()
        st.success("Processed Streamlit cache cleared. FastF1 disk cache remains available.")

if st.session_state["analysis_ready"]:
    df = st.session_state["df"]
    telemetry_data = st.session_state["telemetry_data"]
    selected_section = st.session_state["selected_section"]
    analysis_config = st.session_state["analysis_config"]
    skipped_drivers = st.session_state["skipped_drivers"]
    st.caption(
        f"Loaded: {analysis_config['year']} | {analysis_config['grand_prix']} | "
        f"{analysis_config['session_type']} | Drivers: {len(analysis_config['driver_codes'])} | "
        f"Sector: {analysis_config['start_distance']}-{analysis_config['end_distance']} m"
    )

    if selected_section == "Overview":
        render_overview(df, analysis_config, skipped_drivers, st.session_state["last_load_time_sec"])
    elif selected_section == "Telemetry Trace Analysis":
        render_telemetry_section(df, telemetry_data)
    elif selected_section == "Driver Comparison":
        render_driver_comparison(df)
    elif selected_section == "Cluster Visualization":
        render_cluster_visualization(df)
    elif selected_section == "Cluster Profiles":
        render_cluster_profiles(df)
    elif selected_section == "Driver Profiles":
        render_driver_profiles(df)
    elif selected_section == "Braking Zone Finder":
        render_braking_zone_finder(telemetry_data)
    elif selected_section == "Model Evaluation":
        render_model_evaluation(df, st.session_state["X_scaled"])
    elif selected_section == "Sector Comparison":
        render_sector_comparison(df)
    elif selected_section == "Dataset Export":
        render_dataset_export()
    elif selected_section == "Consistency Analysis":
        render_consistency_analysis()
else:
    st.markdown(
        """
        <div class="empty-state-card">
            Click <b>Run Micro-Sector Analysis</b> in the sidebar to load telemetry data,
            then navigate across sections.
        </div>
        """,
        unsafe_allow_html=True,
    )