import logging
import os

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_ENABLED = False


def enable_fastf1_cache(cache_dir="cache"):
    global _CACHE_ENABLED
    if _CACHE_ENABLED:
        return
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    _CACHE_ENABLED = True


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
    session = fetch_session(year, grand_prix, session_type, cache_dir=cache_dir)
    telemetry_data = {}
    skipped_drivers = []

    for driver_code in driver_codes:
        tel, skip_reason = get_driver_fastest_lap_telemetry(session, driver_code)
        if tel is None:
            skipped_drivers.append(
                {
                    "Driver": driver_code,
                    "Reason": skip_reason or "Telemetry unavailable",
                    "Grand Prix": grand_prix,
                    "Session": session_type,
                }
            )
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
    """Fetch top-5-laps telemetry for all drivers from a single session load."""
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