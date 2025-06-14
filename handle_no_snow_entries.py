# -*- coding: utf-8 -*-
"""
Created on Fri May 16 13:42:03 2025

@author: oconnorb
"""

import pandas as pd
import streamlit as st
import os

def handle_no_snow_entries(df, df_notprocessed, add_to_df_notprocessed, warn_str, study_area, survey_year):
    """
    Handle 'no snow' entries:
    - Remove invalid entries marked 'no snow' but containing data.
    - Generate zero-value rows for depths and densities.
    - Drop original 'no snow' summary row.
    """
    
    # Study Area Dict
    study_area_names = {
    "CRU": "Cruickshank",
    "TSI": "Tsitika",
    "MV": "Metro Vancouver",
    "ENG": "Englishman"
    }
    
    # Step 1: Remove invalid 'no snow' rows with actual values
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

    # Step 2: Load plot features file
    filename = f"{survey_year}_{study_area}_ACO_PlotFeatures.xlsx"
    plot_features_filepath = os.path.join(
        r"S:\ACO\plot_locations", str(survey_year), study_area_names.get(study_area), filename
    )
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
