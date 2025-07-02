import pandas as pd
import utm
import streamlit as st
import numpy as np
import warnings
import zipfile
import io
from project_utils import filter_to_not_processed, process_no_snow_entries_wxform, convert_sample_rating

st.set_page_config(layout="wide")

'''
# CHRL WX Station Visit Form Processing

1. Upload a file (.csv or .xlsx) containing Device Magic Wx Station Visit forms.
2. Upload a GNSS coordinate file for the plots (must include 'plot_id', 'easting_m', 'northing_m').
3. Select the study area and survey date.
4. Click 'Process Forms'.
5. Review warnings and download output.
'''

# === User inputs ===
wxform_file = st.file_uploader('**Upload Device Magic Wx Station Visit Forms**')
gnss_file = st.file_uploader('Upload GNSS Coordinate File')
study_area = st.selectbox("Select Study Area", ('Cruickshank', 'Englishman', 'Metro Vancouver', 'Russell Creek'), index=None)
survey_date = st.date_input("Select Survey Date", value=None, format="YYYY/MM/DD")

study_area_dict = {'Cruickshank': 'CRU', 'Englishman': 'ENG', 'Metro Vancouver': 'MV', 'Russell Creek': 'TSI'}
utm_dict = {'Cruickshank': 10, 'Englishman': 10, 'Metro Vancouver': 10, 'Russell Creek': 9}

target_cols = [
    'Job Start Time', 'Project Name', 'Russell Creek Substation', 'Other Station Name',
    'User', 'Snow Course : Add Snow Core : Depth Max (cm)',
    'Snow Course : Add Snow Core : SWE (cm)', 'Snow Course : Add Snow Core : Density',
    'Snow Course : Add Snow Core : Core Rating', 'Snow Course : Snow Course Notes',
    'Snow Course : Add Snow Core : Snow Core #', 'Snow Course : Add Snow Core : Retrieval (%)',
    'Snow Course : Is There Snow?', 'Submission ID', 'What jobs are being completed? : Snow Course'
]

output_cols = [
    "survey_ID", "start_time", "study_area", "other_study_area", "users",
    "plot_id", "pre_survey_notes", "plot_type", "cardinal_dir", "other_direction",
    "distance_m", "custom_distance_m", "sample_type", "scale_type", "multicore",
    "plot_features", "snow_depth", "swe_final", "density",
    "sample_rating", "obs_notes", "easting_m", "northing_m", "lat", "lon", "flag"
]

if st.button('Process Forms'):
    if wxform_file is None or gnss_file is None:
        st.warning('Both files must be uploaded.')
        st.stop()
    if study_area is None or survey_date is None:
        st.warning('Please select both study area and survey date.')
        st.stop()

    sep, header_row = ',', 0
    first_line = wxform_file.getvalue().decode('utf-8', 'ignore').splitlines()[0]
    if first_line.startswith('sep='):
        sep = first_line.split('=')[1].strip()
        header_row = 1
    wxform_file.seek(0)
    df_raw = pd.read_csv(wxform_file, sep=sep, header=header_row)

    df_raw.columns = df_raw.columns.str.strip()
    target_cols_clean = [c.strip() for c in target_cols]

    present_cols = [c for c in target_cols_clean if c in df_raw.columns]
    missing_cols = [c for c in target_cols_clean if c not in df_raw.columns]

    if missing_cols:
        st.warning(f"Missing expected columns: {missing_cols}")

    df_wxform = df_raw[present_cols]
    initial_length = len(df_wxform)

    df_gnss = pd.read_excel(gnss_file, usecols=['plot_id', 'easting_m', 'northing_m'])
    df_gnss.columns = df_gnss.columns.str.strip()
    if df_gnss['plot_id'].duplicated().any():
        st.error("GNSS file has duplicate plot_id entries.")
        st.stop()

    df_wxform['plot_id'] = df_wxform.apply(
        lambda row: row.get('Other Station Name') if row.get('Russell Creek Substation') == 'Other'
        else row.get('Russell Creek Substation'), axis=1
    )

    df = pd.DataFrame()
    df['start_time'] = df_wxform.get('Job Start Time')
    df['study_area'] = df_wxform.get('Project Name')
    df['users'] = df_wxform.get('User')
    df['is_there_snow'] = df_wxform.get('Snow Course : Is There Snow?')
    df['snow_depth'] = df_wxform.get('Snow Course : Add Snow Core : Depth Max (cm)')
    df['swe_final'] = df_wxform.get('Snow Course : Add Snow Core : SWE (cm)')
    df['density'] = df_wxform.get('Snow Course : Add Snow Core : Density')
    df['sample_rating'] = df_wxform.get('Snow Course : Add Snow Core : Core Rating')
    df['obs_notes'] = df_wxform.get('Snow Course : Snow Course Notes')
    df['retrieval'] = df_wxform.get('Snow Course : Add Snow Core : Retrieval (%)')
    df['core_number'] = df_wxform.get('Snow Course : Add Snow Core : Snow Core #')
    df['plot_id'] = df_wxform['plot_id']
    df['plot_type'] = 'Snow Course'
    df['sample_type'] = 'Density'
    df['scale_type'] = 'Mass'
    df['survey_ID'] = survey_date.strftime("%Y%m%d")
    df['sub_id'] = df_wxform['Submission ID']
    df['snow_course_done'] = df_wxform['What jobs are being completed? : Snow Course']

    survey_date_str = df['survey_ID'].iloc[0]
    study_area_str = study_area_dict[study_area]
    warn_str = f"*{survey_date_str}_{study_area_str}_snowcourse_NOTprocessed.csv*"

    df = df.merge(df_gnss, on='plot_id', how='left')

    df_notprocessed = pd.DataFrame(columns=df.columns)
    df_notprocessed.insert(0, 'problem', [])
    st.session_state.warnings = []
    
    if(df['snow_course_done'] == 'no').any():
        ix = df['snow_course_done'] == 'no'
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'no snow course')
        st.session_state.warnings.append(f'{n_flagged}/{initial_length} entries did not contain a snow course job. Added to {warn_str}')
        
    if (df['study_area'] != study_area).any():
        ix = df['study_area'] != study_area
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'wrong/no study area')
        st.session_state.warnings.append(f'{n_flagged}/{initial_length} entries had the wrong/no study area. Added to {warn_str}')

    df, df_notprocessed = process_no_snow_entries_wxform(
        df, df_notprocessed, filter_to_not_processed, warn_str, study_area_str, str(survey_date.year)
    )

    if df['plot_id'].isna().any():
        ix = df['plot_id'].isna()
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'missing plot_id')
        st.session_state.warnings.append(f'{n_flagged}/{initial_length} entries missing plot_id. Added to {warn_str}')

    if df[['density', 'snow_depth', 'core_number', 'swe_final']].isna().all(axis=1).any():
        ix = df[['density', 'snow_depth', 'core_number', 'swe_final']].isna().all(axis=1)
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'missing snow course data')
        st.session_state.warnings.append(f'{n_flagged}/{initial_length} are missing snow course data. Added to {warn_str}')

    if any(~np.isin(df['plot_id'], df_gnss['plot_id'])):
        ix = ~np.isin(df['plot_id'], df_gnss['plot_id'])
        plot_str = np.unique(df.loc[ix, 'plot_id'])
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'plot_id not in GNSS')
        st.session_state.warnings.append(f'{n_flagged}/{initial_length} entries had unknown plot_id: {", ".join(plot_str)}. Added to {warn_str}')

    if any((df['easting_m'] < 100) | (df['northing_m'] < 100) | df['easting_m'].isnull() | df['northing_m'].isnull()):
        ix = (df['easting_m'] < 100) | (df['northing_m'] < 100) | df['easting_m'].isnull() | df['northing_m'].isnull()
        df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'bad coordinates')
        st.session_state.warnings.append(f'{n_flagged}/{initial_length} entries had bad coordinates. Added to {warn_str}')
        
    # convert all utm coords to lat lon
    df['lat'], df['lon'] = utm.to_latlon(df['easting_m'].values, df['northing_m'].values, utm_dict[study_area], 'U')
    # (Enable above if needed for lat/lon columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_summary = df.groupby('plot_id').agg({
            "snow_depth": ["count", "mean", "median", "std"],
            "density": ["count", "mean", "std"],
            "swe_final": ["mean", "median", "std"],
        }).reset_index()
        df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]

    for col in output_cols:
        if col not in df.columns:
            df[col] = None
            
    # Convert descriptive sample ratings to numeric
    df['sample_rating'] = df['sample_rating'].apply(convert_sample_rating)

    df.loc[df['sample_rating'] < 3, 'flag'] = df['flag'].fillna('') + 'QUALITY'
    df.loc[df['retrieval'] < 60, 'flag'] = df['flag'].fillna('') + ', RETRIEVAL < 60'
    df.loc[df['retrieval'] > 90, 'flag'] = df['flag'].fillna('') + ', RETRIEVAL > 90'

    df = df[output_cols]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "x") as z:
        z.writestr(f"{survey_date_str}_{study_area_str}_snowcourse_processed.csv", df.to_csv(index=False))
        z.writestr(f"{survey_date_str}_{study_area_str}_snowcourse_summarystats.csv", df_summary.to_csv(index=False))
        if len(df_notprocessed):
            df_notprocessed.index += 2
            z.writestr(f"{survey_date_str}_{study_area_str}_snowcourse_NOTprocessed.csv", df_notprocessed.to_csv(index=True, index_label="input_row"))

    if st.session_state.warnings:
        for w in st.session_state.warnings:
            st.warning(w)
    else:
        st.success("All data processed successfully! ✅")

    st.download_button(
        label="Download Output Files",
        data=buf.getvalue(),
        file_name=f"{survey_date_str}_{study_area_str}_snowcourse_outputfiles.zip",
        mime="application/zip"
    )