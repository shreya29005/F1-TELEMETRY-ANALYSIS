import logging
import os

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_ENABLED = False

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "data/results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "race_results.parquet")


def _processed_path(year, grand_prix, session_type):
    safe_gp = grand_prix.replace(" ", "_")
    return os.path.join(PROCESSED_DIR, f"{year}_{safe_gp}_{session_type}.parquet")


def normalize_gp_key(grand_prix):
    """Normalize a Grand Prix name/slug to a comparable key, shared by every
    lookup (telemetry files, session metadata, race results) so a user-facing
    GP label like "Bahrain" reliably matches a stored event slug like
    "Bahrain_Grand_Prix".
    """
    return str(grand_prix).lower().replace(" ", "_").replace("grand_prix", "").strip("_")


def _gp_keys_match(key_a, key_b):
    return bool(key_a) and bool(key_b) and (key_a in key_b or key_b in key_a)


def _find_processed_path(year, grand_prix, session_type):
    if not os.path.exists(PROCESSED_DIR):
        return None
    gp_key = normalize_gp_key(grand_prix)
    for fname in os.listdir(PROCESSED_DIR):
        if not fname.endswith(".parquet"):
            continue
        stem = fname[:-len(".parquet")]
        try:
            file_year_str, rest = stem.split("_", 1)
            file_event_slug, file_session_type = rest.rsplit("_", 1)
        except ValueError:
            continue
        if int(file_year_str) != year or file_session_type != session_type:
            continue
        file_gp_key = normalize_gp_key(file_event_slug)
        if _gp_keys_match(gp_key, file_gp_key):
            return os.path.join(PROCESSED_DIR, fname)
    return None


def _load_processed_telemetry(year, grand_prix, session_type):
    path = _find_processed_path(year, grand_prix, session_type)
    if path is None:
        return None
    return pd.read_parquet(path)


_EMPTY_SESSION_METADATA = {
    "available": False,
    "drivers": [],
    "overall_min_distance": None,
    "overall_max_distance": None,
    "driver_ranges": {},
}


def get_processed_session_metadata(year, grand_prix, session_type):
    """Lightweight metadata (available drivers + telemetry distance ranges) for a
    processed session, without loading the full telemetry columns.
    """
    path = _find_processed_path(year, grand_prix, session_type)
    if path is None:
        return dict(_EMPTY_SESSION_METADATA)

    try:
        meta_df = pd.read_parquet(path, columns=["Driver", "Distance", "LapRank"])
    except Exception:
        try:
            meta_df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"Failed to read processed session metadata from {path}: {exc}")
            return dict(_EMPTY_SESSION_METADATA)

    if "Driver" not in meta_df.columns or "Distance" not in meta_df.columns:
        return dict(_EMPTY_SESSION_METADATA)

    if "LapRank" in meta_df.columns:
        meta_df = meta_df[meta_df["LapRank"] == 1]

    meta_df = meta_df[["Driver", "Distance"]].copy()
    meta_df["Distance"] = pd.to_numeric(meta_df["Distance"], errors="coerce")
    meta_df = meta_df.dropna(subset=["Driver", "Distance"])

    if meta_df.empty:
        return dict(_EMPTY_SESSION_METADATA)

    driver_ranges = {
        driver_code: {"min": float(group["Distance"].min()), "max": float(group["Distance"].max())}
        for driver_code, group in meta_df.groupby("Driver")
    }
    if not driver_ranges:
        return dict(_EMPTY_SESSION_METADATA)

    return {
        "available": True,
        "drivers": sorted(driver_ranges.keys()),
        "overall_min_distance": float(meta_df["Distance"].min()),
        "overall_max_distance": float(meta_df["Distance"].max()),
        "driver_ranges": driver_ranges,
    }


_EMPTY_RACE_PODIUM = {
    "available": False,
    "year": None,
    "grand_prix": None,
    "podium": [],
}


def _resolve_finishing_position(row):
    """Prefer the numeric Position column; fall back to a safely-parsed
    Classified_Position (e.g. "1", "2") and treat non-numeric classifications
    (DNF, DSQ, NC, ...) as unranked rather than guessing a position.
    """
    position = pd.to_numeric(row.get("Position"), errors="coerce")
    if pd.notna(position):
        return float(position)
    classified = row.get("Classified_Position")
    if classified is None or (isinstance(classified, float) and pd.isna(classified)):
        return None
    try:
        return float(int(str(classified).strip()))
    except (TypeError, ValueError):
        return None


def get_official_race_podium(year, grand_prix):
    """Official top-three classified Race result for a season/Grand Prix,
    read from the committed results dataset (data/results/race_results.parquet).
    Never raises — returns a structured "unavailable" result if the dataset is
    missing, the event isn't found, or fewer than three drivers have a valid
    classified finishing position.
    """
    empty_result = dict(_EMPTY_RACE_PODIUM)
    empty_result["year"] = year
    empty_result["grand_prix"] = grand_prix

    if not os.path.exists(RESULTS_PATH):
        return empty_result

    try:
        results_df = pd.read_parquet(RESULTS_PATH)
    except Exception as exc:
        logger.warning(f"Failed to read race results dataset: {exc}")
        return empty_result

    if results_df.empty or "Year" not in results_df.columns or "Grand_Prix" not in results_df.columns:
        return empty_result

    try:
        target_year = int(year)
        year_results = results_df[pd.to_numeric(results_df["Year"], errors="coerce") == target_year]
    except (TypeError, ValueError):
        return empty_result

    if year_results.empty:
        return empty_result

    gp_key = normalize_gp_key(grand_prix)
    matched_event = None
    for candidate_gp in year_results["Grand_Prix"].dropna().unique():
        if _gp_keys_match(gp_key, normalize_gp_key(candidate_gp)):
            matched_event = candidate_gp
            break

    if matched_event is None:
        return empty_result

    event_results = year_results[year_results["Grand_Prix"] == matched_event].copy()
    if event_results.empty:
        return empty_result

    event_results["_ResolvedPosition"] = event_results.apply(_resolve_finishing_position, axis=1)
    event_results = event_results.dropna(subset=["_ResolvedPosition"])
    if len(event_results) < 3:
        return empty_result

    event_results = event_results.sort_values("_ResolvedPosition").head(3)

    podium = []
    for _, row in event_results.iterrows():
        podium.append({
            "Position": int(row["_ResolvedPosition"]),
            "Driver": row.get("Driver"),
            "Driver_Name": row.get("Driver_Name") if pd.notna(row.get("Driver_Name")) else None,
            "Team": row.get("Team") if pd.notna(row.get("Team")) else None,
            "Status": row.get("Status") if pd.notna(row.get("Status")) else None,
            "Points": float(row["Points"]) if pd.notna(row.get("Points")) else None,
        })

    return {
        "available": True,
        "year": target_year,
        "grand_prix": matched_event,
        "podium": podium,
    }
def enable_fastf1_cache(cache_dir="cache"):
    global _CACHE_ENABLED
    if _CACHE_ENABLED:
        return
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    _CACHE_ENABLED = True
enable_fastf1_cache() 

def fetch_session(year, grand_prix, session_type, cache_dir="cache"):
    enable_fastf1_cache(cache_dir=cache_dir)

    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()
    return session


def get_driver_fastest_lap_telemetry(session, driver_code):
    try:
        lap = session.laps.pick_drivers(driver_code).pick_fastest()
        if lap is None:
            logger.warning(f"No valid fastest lap found for driver {driver_code}")
            return None, "No valid fastest lap found"

        tel = lap.get_telemetry()
        if tel is None or tel.empty:
            logger.warning(f"Telemetry unavailable for driver {driver_code}")
            return None, "Telemetry unavailable"
    except Exception as exc:
        logger.warning(f"Failed to fetch telemetry for {driver_code}: {exc}")
        return None, "Telemetry unavailable"

    useful_columns = ["Distance", "Speed", "Throttle", "Brake", "RPM"]
    available_columns = [col for col in useful_columns if col in tel.columns]
    if not available_columns:
        logger.warning(f"Telemetry unavailable for driver {driver_code}: no usable columns")
        return None, "Telemetry unavailable"
    return tel[available_columns].copy(), None

def fetch_driver_telemetry(year, grand_prix, session_type, driver_codes, cache_dir="cache"):
    processed = _load_processed_telemetry(year, grand_prix, session_type)
    if processed is not None:
        telemetry_data = {}
        skipped_drivers = []
        for driver_code in driver_codes:
            driver_rows = processed[(processed["Driver"] == driver_code) & (processed["LapRank"] == 1)]
            if driver_rows.empty:
                skipped_drivers.append({
                    "Driver": driver_code,
                    "Reason": "Telemetry unavailable",
                    "Grand Prix": grand_prix,
                    "Session": session_type,
                })
                continue
            telemetry_data[driver_code] = driver_rows[["Distance", "Speed", "Throttle", "Brake", "RPM"]].reset_index(drop=True)
        return telemetry_data, skipped_drivers

    # fallback: live fetch (works locally; will fail on Cloud if this session wasn't pre-exported)
    session = fetch_session(year, grand_prix, session_type, cache_dir=cache_dir)
    telemetry_data = {}
    skipped_drivers = []
    for driver_code in driver_codes:
        tel, skip_reason = get_driver_fastest_lap_telemetry(session, driver_code)
        if tel is None:
            skipped_drivers.append({"Driver": driver_code, "Reason": skip_reason or "Telemetry unavailable", "Grand Prix": grand_prix, "Session": session_type})
            continue
        telemetry_data[driver_code] = tel
    return telemetry_data, skipped_drivers

def get_driver_top5_laps_telemetry(session, driver_code):
    """Return list of (lap_number, lap_time_sec, tel_df) for up to 5 fastest valid laps."""
    try:
        driver_laps = session.laps.pick_drivers(driver_code)
    except Exception as exc:
        logger.warning(f"Failed to pick laps for {driver_code}: {exc}")
        return [], "Laps unavailable"

    if driver_laps is None or driver_laps.empty:
        return [], "No laps found"

    valid_laps = driver_laps.dropna(subset=["LapTime"]).copy()
    if valid_laps.empty:
        return [], "No valid lap times"

    valid_laps["_LapTime_sec"] = valid_laps["LapTime"].dt.total_seconds()
    valid_laps = valid_laps.dropna(subset=["_LapTime_sec"])
    valid_laps = valid_laps[valid_laps["_LapTime_sec"] > 0]
    valid_laps = valid_laps.sort_values("_LapTime_sec").head(5)

    if valid_laps.empty:
        return [], "No valid laps after filtering"

    useful_columns = ["Distance", "Speed", "Throttle", "Brake", "RPM"]
    result = []
    for _, lap_row in valid_laps.iterrows():
        try:
            tel = lap_row.get_telemetry()
            if tel is None or tel.empty:
                continue
            available_columns = [col for col in useful_columns if col in tel.columns]
            if not available_columns:
                continue
            lap_number_val = lap_row.get("LapNumber")
            lap_number = int(lap_number_val) if pd.notna(lap_number_val) else -1
            lap_time_sec = float(lap_row["_LapTime_sec"])
            result.append((lap_number, lap_time_sec, tel[available_columns].copy()))
        except Exception as exc:
            logger.warning(f"Telemetry error for {driver_code} lap: {exc}")

    if len(result) < 2:
        found = len(result)
        return [], (
            f"Insufficient laps for consistency analysis (found {found}, need at least 2)"
        )
    return result, None


def fetch_driver_top5_telemetry(year, grand_prix, session_type, driver_codes, cache_dir="cache"):
    processed = _load_processed_telemetry(year, grand_prix, session_type)
    if processed is not None:
        top5_data = {}
        skipped = []
        for driver_code in driver_codes:
            driver_rows = processed[processed["Driver"] == driver_code]
            if driver_rows.empty:
                skipped.append({"Driver": driver_code, "Reason": "No valid laps"})
                continue
            laps = []
            for lap_rank, group in driver_rows.groupby("LapRank"):
                lap_number = int(group["LapNumber"].iloc[0])
                lap_time_sec = float(group["LapTime_sec"].iloc[0])
                tel = group[["Distance", "Speed", "Throttle", "Brake", "RPM"]].reset_index(drop=True)
                laps.append((lap_number, lap_time_sec, tel))
            top5_data[driver_code] = laps
        return top5_data, skipped

    # fallback: live fetch
    session = fetch_session(year, grand_prix, session_type, cache_dir=cache_dir)
    top5_data = {}
    skipped = []
    for driver_code in driver_codes:
        laps_data, error = get_driver_top5_laps_telemetry(session, driver_code)
        if not laps_data:
            skipped.append({"Driver": driver_code, "Reason": error or "No valid laps"})
        else:
            top5_data[driver_code] = laps_data
    return top5_data, skipped


if __name__ == "__main__":
    default_drivers = ["VER", "LEC", "HAM"]
    telemetry, skipped = fetch_driver_telemetry(2025, "Bahrain", "Q", default_drivers)
    print(f"Loaded telemetry for: {list(telemetry.keys())}")
    print(f"Skipped: {skipped}")