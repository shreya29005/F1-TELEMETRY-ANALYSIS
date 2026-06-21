import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def minmax_norm(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.empty:
        return pd.Series([], index=series.index, dtype=float)
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((series - series.min()) / (series.max() - series.min())).clip(0.0, 1.0)


def extract_microsector_telemetry(tel, start_distance=300, end_distance=600):
    if tel is None or tel.empty or "Distance" not in tel.columns:
        return pd.DataFrame()
    return tel[(tel["Distance"] >= start_distance) & (tel["Distance"] <= end_distance)].copy()


def extract_turn_features(
    driver_code,
    tel,
    start_distance=300,
    end_distance=600,
    sector_name="Selected sector",
    metadata=None,
):
    if tel is None or tel.empty:
        logger.warning(f"Skipping driver {driver_code}: Telemetry unavailable")
        return None

    t1_tel = extract_microsector_telemetry(tel, start_distance, end_distance)
    if t1_tel is None or t1_tel.empty:
        logger.warning(
            f"Skipping driver {driver_code}: No telemetry points in selected distance window for {sector_name}"
        )
        return None

    if "Speed" not in t1_tel.columns or "Distance" not in t1_tel.columns:
        logger.warning(f"Skipping driver {driver_code}: Required telemetry columns missing")
        return None

    entry_speed = float(t1_tel["Speed"].iloc[0])
    apex_idx = int(t1_tel["Speed"].idxmin())
    apex_speed = float(t1_tel.loc[apex_idx, "Speed"])
    exit_speed = float(t1_tel["Speed"].iloc[-1])
    apex_distance = float(t1_tel.loc[apex_idx, "Distance"])

    avg_speed = float(t1_tel["Speed"].mean())
    speed_std = float(t1_tel["Speed"].std(ddof=0)) if len(t1_tel) > 1 else 0.0

    # Robust brake active detection (works for bool or numeric)
    if "Brake" in t1_tel.columns:
        brake_series = t1_tel["Brake"]
        if pd.api.types.is_bool_dtype(brake_series):
            brake_active = brake_series.fillna(False)
        else:
            brake_active = brake_series.fillna(0) > 0
    else:
        brake_active = None

    brake_start_distance = np.nan
    brake_end_distance = np.nan
    braking_zone_length = np.nan
    trail_braking_index = np.nan
    braking_pct = 0.0

    if brake_active is not None:
        braking_pct = float(brake_active.sum() / len(t1_tel) * 100)
        if brake_active.any():
            brake_distances = t1_tel.loc[brake_active, "Distance"].astype(float)
            brake_start_distance = float(brake_distances.iloc[0])
            brake_end_distance = float(brake_distances.iloc[-1])
            braking_zone_length = float(brake_end_distance - brake_start_distance)

            low_speed_threshold = (entry_speed + apex_speed) / 2.0
            low_speed_mask = t1_tel["Speed"] <= low_speed_threshold
            low_speed_count = int(low_speed_mask.sum())
            if low_speed_count > 0:
                trail_braking_index = float((brake_active & low_speed_mask).sum() / low_speed_count * 100)

    # Throttle derived features
    throttle_pct = 0.0
    throttle_reapply_distance = np.nan
    avg_throttle_after_apex = np.nan
    full_throttle_pct = np.nan
    throttle_commitment_index = np.nan

    if "Throttle" in t1_tel.columns:
        throttle_series = t1_tel["Throttle"].fillna(0).astype(float)
        throttle_pct = float((throttle_series > 50).sum() / len(t1_tel) * 100)
        full_throttle_pct = float((throttle_series == 100).sum() / len(t1_tel) * 100)

        after_apex = t1_tel[t1_tel["Distance"] >= apex_distance].copy()
        if not after_apex.empty:
            after_throttle = after_apex["Throttle"].fillna(0).astype(float)
            avg_throttle_after_apex = float(after_throttle.mean())
            throttle_commitment_index = float((after_throttle > 50).sum() / len(after_apex) * 100)

            reapply_rows = after_apex[after_throttle > 50]
            if not reapply_rows.empty:
                throttle_reapply_distance = float(reapply_rows["Distance"].iloc[0])

    speed_loss = float(entry_speed - apex_speed)
    speed_recovery = float(exit_speed - apex_speed)

    corner_aggression_score = np.nan
    smoothness_index = float(1.0 / max(speed_std, 1e-6))

    feature_metadata = metadata or {}
    features = {
        "Driver": driver_code,
        "Sector_Name": sector_name,
        "Sector_Start": float(start_distance),
        "Sector_End": float(end_distance),
        "Entry_Speed": entry_speed,
        "Apex_Speed": apex_speed,
        "Braking_Pct": braking_pct,
        "Throttle_Pct": throttle_pct,
        "Exit_Speed": exit_speed,
        "Speed_Loss": speed_loss,
        "Speed_Recovery": speed_recovery,
        "Avg_Speed_MicroSector": avg_speed,
        "Speed_Std_MicroSector": speed_std,
        "Brake_Start_Distance": brake_start_distance,
        "Brake_End_Distance": brake_end_distance,
        "Braking_Zone_Length": braking_zone_length,
        "Trail_Braking_Index": trail_braking_index,
        "Throttle_Reapply_Distance": throttle_reapply_distance,
        "Avg_Throttle_After_Apex": avg_throttle_after_apex,
        "Full_Throttle_Pct": full_throttle_pct,
        "Throttle_Commitment_Index": throttle_commitment_index,
        "Corner_Aggression_Score": corner_aggression_score,
        "Smoothness_Index": smoothness_index,
    }
    features.update(feature_metadata)
    return features


def build_feature_dataset(
    telemetry_data,
    start_distance=300,
    end_distance=600,
    sector_definitions=None,
    metadata=None,
):
    if sector_definitions is None:
        sector_definitions = [
            {
                "Sector_Name": "Selected sector",
                "Sector_Start": start_distance,
                "Sector_End": end_distance,
            }
        ]

    normalized_sectors = []
    for sector in sector_definitions:
        if isinstance(sector, dict):
            normalized_sectors.append(
                {
                    "Sector_Name": sector.get("Sector_Name") or "Selected sector",
                    "Sector_Start": sector.get("Sector_Start", start_distance),
                    "Sector_End": sector.get("Sector_End", end_distance),
                }
            )
        elif isinstance(sector, (tuple, list)) and len(sector) >= 3:
            normalized_sectors.append(
                {
                    "Sector_Name": str(sector[0]),
                    "Sector_Start": sector[1],
                    "Sector_End": sector[2],
                }
            )

    features_list = []
    for driver_code, tel in telemetry_data.items():
        for sector in normalized_sectors:
            driver_features = extract_turn_features(
                driver_code,
                tel,
                start_distance=sector["Sector_Start"],
                end_distance=sector["Sector_End"],
                sector_name=sector["Sector_Name"],
                metadata=metadata,
            )
            if driver_features is not None:
                features_list.append(driver_features)

    feature_df = pd.DataFrame(features_list)
    if feature_df.empty:
        return feature_df

    for column in ["Speed_Loss", "Braking_Pct", "Throttle_Commitment_Index"]:
        if column not in feature_df.columns:
            logger.warning(f"Missing {column} when computing normalized aggression score.")
            feature_df[column] = np.nan

        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        column_median = feature_df[column].median()
        if pd.isna(column_median):
            column_median = 0.0
        feature_df[column] = feature_df[column].fillna(column_median)

    feature_df["Speed_Loss_Norm"] = minmax_norm(feature_df["Speed_Loss"])
    feature_df["Braking_Pct_Norm"] = minmax_norm(feature_df["Braking_Pct"])
    feature_df["Throttle_Commitment_Norm"] = minmax_norm(feature_df["Throttle_Commitment_Index"])

    feature_df["Corner_Aggression_Score"] = (
        100.0
        * (
            0.4 * feature_df["Speed_Loss_Norm"]
            + 0.3 * feature_df["Braking_Pct_Norm"]
            + 0.3 * feature_df["Throttle_Commitment_Norm"]
        )
    ).round(2)

    return feature_df