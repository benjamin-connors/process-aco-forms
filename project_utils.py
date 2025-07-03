# -*- coding: utf-8 -*-
"""
Created on Fri May 16 14:04:33 2025

@author: oconnorb
"""

import pandas as pd
import numpy as np
import streamlit as st
import os

def filter_to_not_processed(df, df_notprocessed, mask, problem_label):
    """
    Move rows where mask is True to df_notprocessed with a problem label, 
    and return the cleaned df and updated df_notprocessed.
    """
    if not isinstance(mask, (pd.Series, np.ndarray)) or len(mask) != len(df):
        raise ValueError("Mask must be a boolean Series or array matching df length")

    rows_to_flag = df.loc[mask].copy()
    rows_to_flag['problem'] = problem_label
    df_notprocessed = pd.concat([df_notprocessed, rows_to_flag], ignore_index=True)

    df_cleaned = df.loc[~mask].copy()
    return df_cleaned, df_notprocessed, mask.sum()

def process_no_snow_entries_acoform(df, df_notprocessed, filter_to_not_processed, warn_str, study_area, survey_year):
    """
    Handle 'no snow' entries:
    - Remove invalid entries marked 'no snow' but containing data.
    - Generate zero-value rows for depths and densities.
    - Drop original 'no snow' summary row.
    """
    
    # Step 1: Remove invalid 'no snow' rows with actual values
    no_snow_ix = df["is_there_snow"].str.lower() == "no"
    
    has_real_measurements = (
        (df["depth_cm"].fillna(0).astype(float) > 0) |
        (df["depth_final_cm"].fillna(0).astype(float) > 0) |
        (df["density_gscale"].fillna(0).astype(float) > 0) |
        (df["density_swescale"].fillna(0).astype(float) > 0) |
        (df["swe_final_gscale"].fillna(0).astype(float) > 0) |
        (df["swe_final_swescale"].fillna(0).astype(float) > 0)
    )
    ix = no_snow_ix & has_real_measurements

    if ix.any():
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, "no snow but real measurements entered")
        st.session_state.warnings.append(
            f"Some [{n_flagged}/{len(df)}] entries are marked 'no snow' but contain real measurement data. These entries have been added to {warn_str}."
        )

    # Step 2: Load plot features file using relative path (no subfolders)
    project_root = os.path.dirname(os.path.abspath(__file__))
    plot_features_dir = os.path.join(project_root, "plot_features")
    filename = f"2025_{study_area}_ACO_PlotFeatures.xlsx"
    plot_features_filepath = os.path.join(plot_features_dir, filename)
    
    df_pf = pd.read_excel(
        plot_features_filepath,
        dtype={"plot_id": str, "cardinal_dir": str, "other_direction": str}
    )
    df_pf["distance_m"] = df_pf["distance_m"].astype(float)


    # Step 3: Generate new rows for valid no-snow plots
    cardinal_dirs = ["N", "S", "E", "W"]
    depth_distances = [0.0, 2.5, 5.0, 7.5, 10.0]
    density_distances = [0.0, 10.0]

    no_snow_sub_ids = df[df["is_there_snow"].str.lower() == "no"]["sub_id"].unique()
    generated_rows = []

    for sub_id in no_snow_sub_ids:
        ref_row = df[df["sub_id"] == sub_id].iloc[0].to_dict()

        # Standard 4-direction depths
        for dir in cardinal_dirs:
            for d in depth_distances:
                row = ref_row.copy()
                row.update({
                    "cardinal_dir": dir,
                    "distance_m": d,
                    "sample_type": "Depth",
                    "depth_cm": 0,
                    "depth_final_cm": 0.0,
                    "density_gscale": None,
                    "density_swescale": None,
                    "swe_final_gscale": None,
                    "swe_final_swescale": None,
                    "sample_rating": 5,
                })

                pf_match = df_pf[
                    (df_pf["plot_id"] == row["plot_id"]) &
                    (
                        ((df_pf["cardinal_dir"] == dir) | (df_pf["other_direction"] == dir)) &
                        (df_pf["distance_m"] == d)
                    )
                ]

                if not pf_match.empty:
                    row["plot_features"] = pf_match.iloc[0]["plot_features"]
                    row["other_direction"] = pf_match.iloc[0].get("other_direction", None)

                generated_rows.append(row)

        # Centre depth sample
        row = ref_row.copy()
        row.update({
            "cardinal_dir": "Centre",
            "distance_m": 0.0,
            "sample_type": "Depth",
            "depth_cm": 0,
            "depth_final_cm": 0.0,
            "density_gscale": None,
            "density_swescale": None,
            "swe_final_gscale": None,
            "swe_final_swescale": None,
            "sample_rating": 5,

        })
        generated_rows.append(row)

        # Standard 4-direction densities
        for dir in cardinal_dirs:
            for d in density_distances:
                row = ref_row.copy()
                row.update({
                    "cardinal_dir": dir,
                    "distance_m": d,
                    "sample_type": "Density",
                    "depth_cm": 0,
                    "depth_final_cm": 0.0,
                    "density_gscale": None,
                    "density_swescale": None,
                    "swe_final_gscale": None,
                    "swe_final_swescale": None,
                    "sample_rating": 5,
                })

                if study_area.lower() == "mv":
                    row["density_swescale"] = 0
                    row["swe_final_swescale"] = 0
                else:
                    row["density_gscale"] = 0
                    row["swe_final_gscale"] = 0

                pf_match = df_pf[
                    (df_pf["plot_id"] == row["plot_id"]) &
                    (
                        ((df_pf["cardinal_dir"] == dir) | (df_pf["other_direction"] == dir)) &
                        (df_pf["distance_m"] == d)
                    )
                ]

                if not pf_match.empty:
                    row["plot_features"] = pf_match.iloc[0]["plot_features"]
                    row["other_direction"] = pf_match.iloc[0].get("other_direction", None)

                generated_rows.append(row)

        # Centre density sample
        row = ref_row.copy()
        row.update({
            "cardinal_dir": "Centre",
            "distance_m": 0.0,
            "sample_type": "Density",
            "depth_cm": 0,
            "depth_final_cm": 0.0,
            "density_gscale": None,
            "density_swescale": None,
            "swe_final_gscale": None,
            "swe_final_swescale": None,
            "sample_rating": 5,

        })
        if study_area.lower() == "mv":
            row["density_swescale"] = 0
            row["swe_final_swescale"] = 0
        else:
            row["density_gscale"] = 0
            row["swe_final_gscale"] = 0

        generated_rows.append(row)

    # Step 4: Remove original 'no snow' summary rows
    df = df[df["sub_id"].isin(no_snow_sub_ids) == False]

    # Step 5: Add new rows
    df = pd.concat([df, pd.DataFrame(generated_rows)], ignore_index=True)

    return df, df_notprocessed

def process_no_snow_entries_wxform(df, df_notprocessed, filter_to_not_processed, warn_str, study_area, survey_year):
    """
    Handle 'no snow' entries in CHRL Wx Station Visit forms:
    - Remove invalid 'no snow' rows with actual data.
    - Generate 10 zero-value rows (core_number 1–10) with plot_features='O'.
    - Drop original summary 'no snow' row.
    """

    # Step 1: Remove 'no snow' rows that still contain real data
    no_snow_ix = df["is_there_snow"].str.lower() == "no"
    has_real_measurements = (
        (df["snow_depth"].fillna(0) > 0) |
        (df["density"].fillna(0) > 0) |
        (df["swe_final"].fillna(0) > 0)
    )
    ix = no_snow_ix & has_real_measurements

    if ix.any():
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, "no snow but real measurements entered")
        st.session_state.warnings.append(
            f"Some [{n_flagged}/{len(df)}] entries are marked 'no snow' but contain real measurement data. These entries have been added to {warn_str}."
        )

    # Step 2: Generate 10 dummy rows per no-snow entry
    no_snow_sub_ids = df[df["is_there_snow"].str.lower() == "no"]["sub_id"].unique()
    generated_rows = []

    for sub_id in no_snow_sub_ids:
        ref_row = df[df["sub_id"] == sub_id].iloc[0].to_dict()

        for core_num in range(1, 11):  # Core numbers 1 to 10
            row = ref_row.copy()
            row.update({
                "core_number": core_num,
                "snow_depth": 0,
                "density": 0,
                "swe_final": 0,
                "sample_rating": 5,
            })
            generated_rows.append(row)

    # Step 3: Remove the original summary no-snow rows
    df = df[~df["sub_id"].isin(no_snow_sub_ids)]

    # Step 4: Add the generated rows
    df = pd.concat([df, pd.DataFrame(generated_rows)], ignore_index=True)

    return df, df_notprocessed

def convert_sample_rating(value):
    """
    Convert sample rating strings to numeric values.
    Handles both descriptive words and numeric strings.
    
    Returns:
        int/float/NaN: Numeric equivalent of the rating, or np.nan if unrecognized.
    """
    rating_map = {
        'Excellent': 5,
        'Good': 4,
        'Fair': 3,
        'Poor': 2,
        'Bad': 1
    }

    if isinstance(value, str):
        value = value.strip()
        if value in rating_map:
            return rating_map[value]
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value
