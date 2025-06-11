# -*- coding: utf-8 -*-
"""
Created on Fri May 16 14:04:33 2025

@author: oconnorb
"""

import pandas as pd
import streamlit as st

def add_to_df_notprocessed(df_main: pd.DataFrame,
                     df_notprocessed: pd.DataFrame, 
                     index: pd.Series,
                     problem: str) -> pd.DataFrame:
    """
    Adds problematic rows from the main dataframe to a separate dataframe 
    for tracking entries that were excluded from processing.

    Args:
        df_main (pd.DataFrame): The main DataFrame containing all entries.
        df_notprocessed (pd.DataFrame): The DataFrame where problematic rows are collected.
        index (pd.Series): A boolean Series indicating which rows in df_main to extract.
        problem (str): A short description of the issue, added to a new 'problem' column.

    Returns:
        pd.DataFrame: An updated version of df_notprocessed with the new rows appended.

    Raises:
        ValueError: If the index length does not match the number of rows in df_main.
    """
    if len(index) != len(df_main):
        raise ValueError("Boolean index length must match the length of df_main")

    # Extract the rows that match the problem condition
    rows_to_add = df_main[index].copy()
    rows_to_add['problem'] = problem

    # Append them to the df_notprocessed tracking DataFrame
    df_notprocessed = pd.concat([df_notprocessed, rows_to_add], ignore_index=True)

    return df_notprocessed



def process_no_snow_entries(df, add_to_df_notprocessed, warn_str, plot_features_filepath):
    """
    Validate and process 'no snow' entries:
    - Remove entries marked 'no snow' but have real measurements
    - Generate zero-value depth and density rows for those no-snow plots,
      filling plot_features and directions from external reference Excel
    
    Args:
        df (pd.DataFrame): Main data frame to process.
        add_to_df_notprocessed (function): Callback to register excluded rows.
        warn_str (str): Warning message suffix for logging.
        plot_features_filepath (str): Path to plot features Excel file.
        
    Returns:
        pd.DataFrame: DataFrame with invalid rows removed and zero-measurement
                      rows added for 'no snow' plots.
    """
    # Step 1: Validate 'no snow' entries with real measurements
    no_snow_ix = df["is_there_snow"].str.lower() == "no"

    has_real_measurements = (
        (df["depth_cm"].fillna(0) > 0) |
        (df["depth_final_cm"].fillna(0) > 0) |
        (df["density_gscale"].fillna(0) > 0) |
        (df["density_swescale"].fillna(0) > 0) |
        (df["swe_final_gscale"].fillna(0) > 0) |
        (df["swe_final_swescale"].fillna(0) > 0)
    )

    ix = no_snow_ix & has_real_measurements

    if ix.any():
        add_to_df_notprocessed(ix, "no snow but real measurements entered")
        st.session_state.warnings.append(
            f"Some [{ix.sum()}/{len(df)}] entries are marked 'no snow' but contain real measurement data. These entries have been added to {warn_str}."
        )
        df = df.loc[~ix]

    # Step 2: Load plot features reference table
    plot_features_df = pd.read_excel(
        plot_features_filepath,
        dtype={"plot_id": str, "cardinal_dir": str, "other_direction": str}
    )
    plot_features_df["distance_m"] = plot_features_df["distance_m"].astype(float)

    # Step 3: Generate zero measurements for no-snow plots
    cardinal_dirs = ["N", "S", "E", "W"]
    depth_distances = [0.0, 2.5, 5.0, 7.5, 10.0]
    density_distances = [0.0, 10.0]

    no_snow_sub_ids = df[df["is_there_snow"].str.lower() == "no"]["sub_id"].unique()

    generated_rows = []

    for sub_id in no_snow_sub_ids:
        ref_row = df[df["sub_id"] == sub_id].iloc[0].to_dict()

        for dir in cardinal_dirs:
            # Depth samples
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
                })

                pf_match = plot_features_df[
                    (plot_features_df["plot_id"] == row["plot_id"]) &
                    (plot_features_df["cardinal_dir"] == dir) &
                    (plot_features_df["distance_m"] == d)
                ]

                if pf_match.empty:
                    pf_match = plot_features_df[
                        (plot_features_df["plot_id"] == row["plot_id"]) &
                        (plot_features_df["other_direction"] == dir) &
                        (plot_features_df["distance_m"] == d)
                    ]

                if not pf_match.empty:
                    row["plot_features"] = pf_match.iloc[0]["plot_features"]
                    row["other_direction"] = pf_match.iloc[0].get("other_direction", None)

                generated_rows.append(row)

            # Density samples
            for d in density_distances:
                row = ref_row.copy()
                row.update({
                    "cardinal_dir": dir,
                    "distance_m": d,
                    "sample_type": "Density",
                    "depth_cm": 0,
                    "depth_final_cm": 0.0,
                    "density_gscale": 0,
                    "density_swescale": 0,
                    "swe_final_gscale": 0,
                    "swe_final_swescale": 0,
                })

                pf_match = plot_features_df[
                    (plot_features_df["plot_id"] == row["plot_id"]) &
                    (plot_features_df["cardinal_dir"] == dir) &
                    (plot_features_df["distance_m"] == d)
                ]

                if pf_match.empty:
                    pf_match = plot_features_df[
                        (plot_features_df["plot_id"] == row["plot_id"]) &
                        (plot_features_df["other_direction"] == dir) &
                        (plot_features_df["distance_m"] == d)
                    ]

                if not pf_match.empty:
                    row["plot_features"] = pf_match.iloc[0]["plot_features"]
                    row["other_direction"] = pf_match.iloc[0].get("other_direction", None)

                generated_rows.append(row)

    # Append generated rows to main dataframe
    new_rows_df = pd.DataFrame(generated_rows)
    df = pd.concat([df, new_rows_df], ignore_index=True)

    return df
