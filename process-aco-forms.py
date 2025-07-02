import numpy as np
import pandas as pd
import utm
import warnings
import matplotlib.pyplot as plt
import io
import zipfile
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.colors as pc
from project_utils import process_no_snow_entries_acoform, filter_to_not_processed, convert_sample_rating

### STREAMLIT LAYOUT ###
st.set_page_config(
    layout="wide")

'''
# CHRL ACO Survey Form Processing

1. In Box 1, upload a file (.csv or .xlsx) containing the Device Magic ACO survey forms to be processed.
2. In Box 2, upload a file (.csv or .xlsx) containing the GNSS coordinates for each of the plots within the survey.  \n **Important:** GNSS file MUST contain the following headers: 'plot_id', 'easting_m', 'northing_m'
3. Select the study area and the survey date for the forms to be processed.
4. Click the 'Process Forms' button.
5. Review any warnings and click the "Download Output Files" button.
'''
# Inputs
dmform_file = st.file_uploader('**Upload File (.xlsx, .csv) Containing Device Magic ACO Survey Forms Here**')
gnss_file = st.file_uploader('Upload File (.xlsx, .csv) Containing GNSS Info Here')
study_area = st.selectbox("Select Study Area", ('Cruickshank', 'Englishman', 'Metro Vancouver', 'Russell Creek'), index=None)
survey_date = st.date_input("Select Survey Date", value=None, format="YYYY/MM/DD")

study_area_dict = {
    'Cruickshank': 'CRU',
    'Englishman': 'ENG',
    'Metro Vancouver': 'MV',
    'Russell Creek': 'TSI'
}
utm_dict = {
    'Cruickshank': 10,
    'Englishman': 10,
    'Metro Vancouver': 10,
    'Russell Creek': 9
}

mandatory_fields = ['study_area', 'plot_type', 'cardinal_dir', 'distance_m']

### MAIN PROGRAM PROCESSING ###
if st.button('Process Forms'):
    # check for input files
    if dmform_file is None or gnss_file is None:
        st.warning('Both files must be uploaded to process the forms.')
        st.session_state.data_processed = False
    elif study_area is None:
        st.warning('Please enter the study area')
    elif survey_date is None:
        st.warning('Please enter the survey date.')
    else:
        # initialize session state variables
        st.session_state.data_processed = False
        st.session_state.warnings = []
        st.session_state.df = None
        st.session_state.df_notprocessed = None
        st.session_state.map = None
        
        # get filenames in lowercase for consistency
        dmform_filename = dmform_file.name.lower()
        gnss_filename = gnss_file.name.lower()

        # Define required columns
        output_cols = [
            "survey_ID", "start_time", "study_area", "other_study_area", "users", 
            "plot_id", "pre_survey_notes", "plot_type", "cardinal_dir", "other_direction", 
            "distance_m", "custom_distance_m", "sample_type", "scale_type", "multicore", 
            "plot_features", "snow_depth", "swe_final", "density",
            "sample_rating", "retrieval", "obs_notes", "easting_m", "northing_m", "lat", "lon", "flag"
        ]

       # Read dmform file, get column names, get number of rows
        if dmform_filename.endswith(".csv"):
            # Decode the binary stream
            first_line = dmform_file.readline().decode("utf-8")
            dmform_file.seek(0)  # Reset pointer to start
        
            if first_line.startswith("sep="):
                separator = first_line.split("=")[1].strip()
                df_dmform = pd.read_csv(dmform_file, sep=separator, skiprows=1)
            else:
                df_dmform = pd.read_csv(dmform_file)
        elif dmform_filename.endswith(".xlsx"):
            df_dmform = pd.read_excel(dmform_file)
                    
        dmform_cols = df_dmform.columns
        initial_length = len(df_dmform)
        
        # get survey_date (first date of survey data)generate warning string
        study_area_str = study_area_dict[study_area]
        survey_date_str = survey_date.strftime("%Y%m%d")
        warn_str = '*' + survey_date_str + '_' + study_area_str+ '_NOTprocessed.csv*'

        # read gnss file and check for repeat plot_ids
        if gnss_filename.endswith(".csv"):
            df_gnss = pd.read_csv(gnss_file, usecols=['plot_id', 'easting_m', 'northing_m', 'color_code'])
        elif gnss_filename.endswith(".xlsx"):
            df_gnss = pd.read_excel(gnss_file, usecols=['plot_id', 'easting_m', 'northing_m', 'color_code'])

        # Remove rows with empty plot_ids
        df_gnss = df_gnss.dropna(subset=['plot_id'])

        if df_gnss['plot_id'].duplicated().any():
            st.error('**ERROR:** The provided GNSS file contains multiple entries for at least one plot ID. Please update the GNSS file so that it contains at most 1 set of coordinates for each plot ID.')
            st.stop()
        # read in fieldnames spreadsheet (contains current and historical DMform names for both .csv and google sheets, flags for which fields to read, and new column names for output)
        df_fieldnames = pd.read_csv('aco_form_fieldnames.csv')

        # find fieldnames to keep based on 'read_flag' columns of fieldnames file
        ix_keep = df_fieldnames['read_flag'].apply(lambda x: True if x == 1 else False)

        # initialize processed and notprocessed dataframes with desired fields (add aco flight)
        df = pd.DataFrame(columns=df_fieldnames['post_process'][ix_keep])

        # loop fields that we want to keep
        for ii in df_fieldnames['post_process'][ix_keep].index:
            # Check list of possible fieldnames to see if one matches the current file
            possible_cols = df_fieldnames.iloc[ii, np.isin(df_fieldnames.iloc[ii, :], dmform_cols)]
            possible_cols = np.unique(possible_cols[possible_cols.isin(df_dmform.columns)])

            if len(possible_cols) > 0:
                # Check which of these columns contain data
                cols_with_data = [col for col in possible_cols if not df_dmform[col].isnull().all()]
                if len(cols_with_data) == 1:
                    # If only one column contains data, use that column
                    df[df_fieldnames['post_process'][ii]] = df_dmform[cols_with_data[0]]
                elif len(cols_with_data) > 1:
                    # If more than one column contains data, check if fields are identical
                    reference_col = df_dmform[cols_with_data[0]]
                    identical = all(reference_col.fillna(df_dmform[col]).equals(df_dmform[col].fillna(reference_col)) for col in cols_with_data)
                    
                    if identical:
                        # All columns are identical where they have values, choose the column with the most data
                        most_data_col = max(cols_with_data, key=lambda col: df_dmform[col].notna().sum())
                        df[df_fieldnames['post_process'][ii]] = df_dmform[most_data_col]
                    else:
                        # Columns are not identical
                        st.error(f'**ERROR:** Multiple columns on the input file {str(cols_with_data)} contain data for the field {df_fieldnames["post_process"][ii]}.')
                        st.stop()
            elif np.isin(df_fieldnames['post_process'][ii], mandatory_fields):
                # If none of the columns contain data, you may want to handle this case
                st.error(f'**ERROR:** No columns found on the input file that contain data for the field {df_fieldnames["post_process"][ii]}.')
                st.stop()
                                                
        # get plot id's
        id_cols = ['plot_id', 'plot_id_tsi', 'plot_id_cru', 'plot_id_eng', 'plot_id_mv',
                   'plot_id_tsi_other', 'plot_id_cru_other', 'plot_id_eng_other', 'plot_id_mv_other']
        fallback_cols = id_cols[1:]

        # Normalize 'Other' to NaN in all relevant columns
        for col in id_cols:
            df[col] = df[col].replace(r'(?i)^other$', np.nan, regex=True)  # case-insensitive replace
        # Count how many valid (non-null) values per row across id_cols
        valid_counts = df[id_cols].notna().sum(axis=1)
        # Raise error only if multiple valid values exist in the same row
        if (valid_counts > 1).any():
            raise ValueError("Multiple plot_id values found in the same row.")
        # Backfill plot_id with the first available fallback value
        df['plot_id'] = df[['plot_id'] + fallback_cols].bfill(axis=1).iloc[:, 0]
        # Drop fallback columns after filling plot_id
        df.drop(columns=fallback_cols, inplace=True)

        # enforce formatting
        df['distance_m'] = pd.to_numeric(df['distance_m'], errors='coerce')
                    
        # add survey date col to df 
        df.insert(0, 'survey_ID', survey_date_str)
        # create not processed df
        df_notprocessed = pd.DataFrame(columns=df.columns)
        df_notprocessed.insert(0,'problem', [])

        # only keep entries for selected study area
        if (df['study_area'] != study_area).any():
            ix = df['study_area'] != study_area
            df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'wrong/no study area')
            st.session_state.warnings.append('Some [' + str(n_flagged) + '/' + str(initial_length) + '] entries have the wrong/no study area. These entries have been added to' + warn_str)
            
        # generate 'zero' measurements for 'no snow' plots
        df, df_notprocessed = process_no_snow_entries_acoform(
            df,
            df_notprocessed,
            filter_to_not_processed,
            warn_str,
            study_area_str,
            str(survey_date.year)
        )
            
        # populate snow_depth (== depth values from both density and depth surveys combined in one field)
        snow_depth = np.maximum(df['depth_final_cm'].fillna(-np.inf), df['depth_max'].fillna(-np.inf))
        snow_depth = pd.Series(snow_depth).replace(-np.inf, np.nan)  
        df.insert(df.columns.get_loc('multicore'), 'snow_depth', snow_depth)
        
        # populate density and SWE (take measurements from both mass scales and swe scales)
        swe_gscale_populated = (df['sample_type'] == 'Density') & df['swe_final_gscale'].notna()
        swe_swescale_populated = (df['sample_type'] == 'Density') & df['swe_final_swescale'].notna()
        density_gscale_populated = (df['sample_type'] == 'Density') & df['density_gscale'].notna()
        density_swescale_populated = (df['sample_type'] == 'Density') & df['density_swescale'].notna()
        swe_both_populated = swe_gscale_populated & swe_swescale_populated
        density_both_populated = density_gscale_populated & density_swescale_populated
        
        # Initialize empty columns
        df['swe_final'] = pd.Series(dtype='float64')
        df['density'] = pd.Series(dtype='float64')
        df['scale_type'] = pd.Series(dtype='string')

        # Fill only where one column is populated
        df.loc[swe_gscale_populated & ~swe_both_populated, 'swe_final'] = df['swe_final_gscale']
        df.loc[swe_gscale_populated & ~swe_both_populated, 'scale_type'] = 'Mass'
        df.loc[swe_swescale_populated & ~swe_both_populated, 'swe_final'] = df['swe_final_swescale']
        df.loc[swe_swescale_populated & ~swe_both_populated, 'scale_type'] = 'SWE'

        df.loc[density_gscale_populated & ~density_both_populated, 'density'] = df['density_gscale']
        df.loc[density_swescale_populated & ~density_both_populated, 'density'] = df['density_swescale']
        
        # add any rows where both mass-scale and swe-scale measurements exist to not processed
        if any(swe_both_populated | density_both_populated):
            ix = swe_both_populated | density_both_populated
            df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'contains both mass-scale and swe-scale measurements')
            st.session_state.warnings.append(
                'Some [' + str(n_flagged) + '/' + str(initial_length) + '] entries contain both mass-scale and swe-scale measurements. '
                'These entries have been added to ' + warn_str)
            
        # add distance_to_centre for centre plots (sometimes missing)
        df.loc[df['cardinal_dir'] == 'Centre', 'distance_m'] = 0

        # Keep only rows that have both a direction (cardinal_dir or other_direction) and a distance (distance_m or custom_distance_m)
        if any(pd.isnull(df['cardinal_dir']) & pd.isnull(df['other_direction']) | 
            pd.isnull(df['distance_m']) & pd.isnull(df['custom_distance_m'])):
            
            ix = (pd.isnull(df['cardinal_dir']) & pd.isnull(df['other_direction'])) | \
                (pd.isnull(df['distance_m']) & pd.isnull(df['custom_distance_m']))
                            
            plot_str = np.unique(df.loc[ix, 'plot_id'])
            df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'missing direction or distance data')
            st.session_state.warnings.append(
                'Some [' + str(n_flagged) + '/' + str(initial_length) + '] entries do not contain both required direction and distance data. '
                'These entries have been added to ' + warn_str + ' for the following plots: (' + ", ".join(plot_str) + ').'
            )

        # keep only rows that have coordinates in gnss file
        if any(~np.isin(df['plot_id'], df_gnss['plot_id'])):
            ix = ~np.isin(df['plot_id'], df_gnss['plot_id'])
            plot_str = np.unique(df.loc[ix, 'plot_id'])
            df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'plot_id not in gnss')
            st.session_state.warnings.append('Some [' + str(n_flagged) + '/' + str(initial_length) + '] entries were not provided coordinates in the GNSS file [' +
                ", ".join(plot_str) + ']. These entries have been added to ' + warn_str)

        # add eastings and northings according to plot id
        df = df.merge(df_gnss, on='plot_id', how='left')
        
        # adjust eastings and northings according to sample distance from centre
        # define dictionary containing angles for each cardinal direction
        cardinal_ang = {
            'E': 0,
            'NE': 45,
            'N': 90,
            'NW': 135,
            'W': 180,
            'SW': 225,
            'S': 270,
            'SE': 315,
            'Centre': 0}
        
        #  missing distance to centre
        if any(np.isnan(df['distance_m'])):
            ix = np.isnan(df['distance_m'])
            df.loc[ix, ['easting_m', 'northing_m']] = None
            df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'missing distance')
            st.session_state.warnings.append('Some [' + str(n_flagged) + '/' + str(initial_length) + '] entries are missing cardinal plot distance. These entries have been added to ' + warn_str)

        # get angles and calc coordinates for each data point
        ang = np.where(
            pd.notna(df['other_direction']), 
            df['other_direction'], 
            df["cardinal_dir"].apply(lambda x: cardinal_ang.get(x))
        )
        
        ang = np.asarray(ang, dtype=float)
        df['easting_m'] = df['easting_m'] + np.round(np.cos(np.deg2rad(ang)), 3) * df['distance_m']
        df['northing_m'] = df['northing_m'] + np.round(np.sin(np.deg2rad(ang)), 3) * df['distance_m']

        # remove nonsense/zero-coordinates
        if any((df['easting_m'] < 100) | (df['northing_m'] < 100) | (df['easting_m'].isnull()) | (df['northing_m'].isnull())):
            ix = (df['easting_m'] < 100) | (df['northing_m'] < 100) | (pd.isnull(df['easting_m'])) | (pd.isnull(df['northing_m']))
            df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'bad coordinates')
            st.session_state.warnings.append('Some [' + str(n_flagged) + '/' + str(initial_length) + '] entries were given bad coordinates (check GNSS file). These entries have been added to ' + warn_str)

        # get lat/lons
        (df['lat'], df['lon']) = utm.to_latlon(df['easting_m'], df['northing_m'], utm_dict[study_area], 'U')

        ### DATA/PROCESSING CHECKS AND DF CLEANUP
        # missing snow_depth
        if any(np.isnan(df['snow_depth'])):
            ix = np.isnan(df['snow_depth'])
            df, df_notprocessed, n_flagged = filter_to_not_processed(df, df_notprocessed, ix, 'missing snow depth')
            st.session_state.warnings.append('Some [' + str(n_flagged) + '/' + str(initial_length) + '] entries are missing snow depth. These entries have been added to ' + warn_str)
            
        # Convert descriptive sample ratings to numeric
        df['sample_rating'] = df['sample_rating'].apply(convert_sample_rating)
        
        # Add missing columns to df (empty)
        for col in output_cols:
            if col not in df.columns:
                df[col] = None
                
        # Remove unwanted columns
        df = df[output_cols]
        
        # Add flags
        df.loc[df['sample_rating'] < 3, 'flag'] = df['flag'].fillna('') + 'QUALITY'
        df.loc[df['multicore'] == 'yes', 'flag'] = df.apply(
            lambda row: row['flag'] + ', MULTI' if pd.notna(row['flag']) else 'MULTI', axis=1)
        
        df.loc[df['retrieval'] < 60, 'flag'] = df.apply(
            lambda row: row['flag'] + ', RETRIEVAL < 60' if pd.notna(row['flag']) else 'RETRIEVAL < 60', axis=1)
        df.loc[df['retrieval'] > 90, 'flag'] = df.apply(
            lambda row: row['flag'] + ', RETRIEVAL > 90' if pd.notna(row['flag']) else 'RETRIEVAL > 90', axis=1)

        # update session state variables
        st.session_state.data_processed = True
        st.session_state.df = df
        st.session_state.df_notprocessed = df_notprocessed

    if st.session_state.data_processed:
        # Print Warning Messages
        if st.session_state.warnings == []:
            st.warning(':green[All data was processed successfully!] 👍')
        else:
            st.warning('**WARNING:** Not all entries were processed successfully [' + str(len(df)) + '/' + str(initial_length) + ' processed, '
                       + str(len(df_notprocessed)) + '/' + str(initial_length) + ' not processed]. See warning messages below.')

            for ii, warn_str in enumerate(st.session_state.warnings):
                st.warning('Warning ' + str(ii+1) + ': ' + str(warn_str))

        # HAS DATA FALLEN THROUGH THE GAPS?
        if (len(df) + len(df_notprocessed)) < initial_length:
            st.warning(":red[**WARNING: DATA HAS FALLEN INTO THE VOID!**\
                \n There is input data that did not make it into either the processed or not-processed output files.\
                \n In other words, some of the data has gone missing.\
                    \n You can still download the data that was processed, but you have been warned that it is incomplete!]")
                       
        # Print multicore warning and notes
        multicore_rows = df[df['multicore'].str.lower() == 'yes']
        if not multicore_rows.empty:
            st.warning("⚠️ The processed file contains multicore samples. Please review the observation notes below and ensure that any necessary manual edits have been applied. Row numbers corresponds to row in processed file.")
            for idx, note in multicore_rows['obs_notes'].items():
                st.markdown(f"**Row {idx}:** {note}")
                       
        # calculate summary statistics dataframe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Calculate summary stats without filling or removing NaN values
            df_summary = df.groupby('plot_id').agg({
                "snow_depth": ["count", "mean", "median", "std"],
                "density": ["count", "mean", "std"],
                "swe_final": ["mean", "median", "std"],
            }).reset_index()
            # Rename columns for better readability
            df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns.values]
        
        # zip output files using buffer memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "x") as csv_zip:
            csv_zip.writestr(survey_date_str + '_' + study_area_str + '_snowsurvey_processed.csv', pd.DataFrame(df).to_csv(index=False))
            csv_zip.writestr(survey_date_str + '_' +  study_area_str + '_snowsurvey_summarystats.csv', pd.DataFrame(df_summary).to_csv(index=False))
            if len(df_notprocessed) > 0:
                # bump index by 2 to align with input form rows
                df_notprocessed.index += 2
                csv_zip.writestr(survey_date_str + '_' +  study_area_str + '_snowsurvey_NOTprocessed.csv', pd.DataFrame(df_notprocessed).to_csv(index=True, index_label='input_row'))

        # download button
        st.download_button(
            label="Download Output Files",
            data=buf.getvalue(),
            file_name=survey_date_str + '_' +  study_area_str + '_snowsurvey_outputfiles.zip',
            mime="application/zip",
            )
    
        ## MAPPING
        # Create a Folium map centered on samples
        min_lat, max_lat = df['lat'].min(), df['lat'].max()
        min_lon, max_lon = df['lon'].min(), df['lon'].max()
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # Add satellite imagery tile layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles &copy; Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            name='Esri World Imagery',
            overlay=False,
            control=True
        ).add_to(m)

        # add a box around sample points
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        folium.Rectangle(bounds=bounds, color='red', fill=False).add_to(m)

        # Calculate mean values for snow_depth and mean of density fields for each plot_id
        mean_values = df.groupby('plot_id').agg({
            'snow_depth': 'mean',
            'density': 'mean',
        }).reset_index()

        # Merge the mean values back to the cleaned DataFrame
        df = df.merge(mean_values, on='plot_id', suffixes=('', '_mean'))

       # Determine unique combinations of 'plot_id', 'cardinal_dir', and 'distance_m'
        unique_combinations = df.drop_duplicates(subset=['plot_id', 'cardinal_dir', 'distance_m'])

        # Get color_code per plot_id from df_gnss
        plot_colors = dict(df_gnss[['plot_id', 'color_code']].dropna().values)
        used_plot_ids = df['plot_id'].unique()
        missing_plot_ids = [pid for pid in used_plot_ids if pid not in plot_colors]
        fallback_color = '#999999'  # a light gray fallback color
        for pid in missing_plot_ids:
            plot_colors[pid] = fallback_color
        if missing_plot_ids:
            st.warning(
                f"⚠️ The following plot IDs do not have a specified color in `df_gnss` "
                f"and have been assigned a fallback color ({fallback_color}):\n\n"
                f"{', '.join(missing_plot_ids)}"
            )

        # Add marker for sample locations, with pop-up showing plotid/depth/density
        radius = 30
        for _, row in unique_combinations.iterrows():
            plot_id = row['plot_id']
            color = plot_colors.get(plot_id, 'red')  # Default to red if plot_id is not found in the dictionary
            folium.Circle(
                location=[row['lat'], row['lon']],
                radius=radius,
                color='white',  # Border color
                weight=1,
                fill_opacity=1,
                opacity=1,
                fill_color=plot_colors[row['plot_id']],  # Fill color
                fill=True,
                popup=row['plot_id'],  # Assuming 'plot_id' is the column with unique identifiers
                tooltip=(
                    f"Plot ID: {row['plot_id']}<br>"
                    f"Mean Snow Depth: {row['snow_depth_mean']:.2f} m<br>"
                    f"Mean Density: {row['density']:.2f}"
                ),
            ).add_to(m)
        # add map as session state variable
        st.session_state.map = m

        # Create histograms for each plot_id
        unique_plot_ids = df['plot_id'].unique()
        color_list = [plot_colors.get(plot_id, '#000000') for plot_id in unique_plot_ids]  # Default to black if plot_id is not found

        # Snow Depth Bar Stacked Plot
        fig1, ax1 = plt.subplots(figsize=(6.5, 3), constrained_layout=True)  # Snow depth histogram        
        bin_edges = np.histogram_bin_edges(df['snow_depth'].dropna(), bins=30)
        
        bottom_stack = np.zeros(len(bin_edges) - 1)  # Initialize stacking baseline
        
        for plot_id, color in zip(unique_plot_ids, color_list):
            subset = df[df['plot_id'] == plot_id]
            hist, _ = np.histogram(subset['snow_depth'].dropna(), bins=bin_edges)
            
            ax1.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', 
                    color=color, alpha=0.7, label=f'Plot {plot_id}', bottom=bottom_stack)
            
            bottom_stack += hist  # Update stack for the next set of bars
        
        ax1.set_title('Bar Stacked Histogram of Snow Depth')
        ax1.set_xlabel('Snow Depth (cm)')
        ax1.set_ylabel('Count')
        
        # Density Bar Stacked Plot
        fig2, ax2 = plt.subplots(figsize=(6.5, 3), constrained_layout=True)  # Snow depth histogram        
        bin_edges_density = np.arange(0, 1.05, 0.025)
        
        bottom_density_stack = np.zeros(len(bin_edges_density) - 1)  # Initialize stacking baseline
        
        for plot_id, color in zip(unique_plot_ids, color_list):
            subset = df[df['plot_id'] == plot_id]
            hist_density, _ = np.histogram(subset['density'].dropna(), bins=bin_edges_density)
            
            ax2.bar(bin_edges_density[:-1], hist_density, width=np.diff(bin_edges_density), 
                    edgecolor='black', color=color, alpha=0.7, label=f'Plot {plot_id}', 
                    bottom=bottom_density_stack)
            
            bottom_density_stack += hist_density  # Update stacking height
        
        ax2.set_title('Bar Stacked Histogram of Density')
        ax2.set_xlabel('Density')
        ax2.set_ylabel('Count')
        
        # --- Snow Depth vs SWE Scatterplot ---
        fig3, ax3 = plt.subplots(figsize=(6.5, 3), constrained_layout=True)  # Snow depth histogram        
    
        # Prepare scatter data
        for plot_id, color in zip(unique_plot_ids, color_list):
            subset = df[(df['plot_id'] == plot_id) & df['snow_depth'].notna() & df['swe_final'].notna()]
            ax3.scatter(
                subset['snow_depth'], subset['swe_final'],
                label=f'Plot {plot_id}', color=color, alpha=0.7, edgecolor='black', s=40
            )
    
        # Fit regression line (y = m·x, no intercept)
        valid = df[['snow_depth', 'swe_final']].dropna()
        X = valid[['snow_depth']].values
        y = valid['swe_final'].values
    
        if len(X) > 1:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
    
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            slope = float(model.coef_[0])
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
    
            # Draw regression line
            x_line = np.linspace(X.min(), X.max(), 100)
            ax3.plot(x_line, slope * x_line, color='black', linewidth=2, linestyle='--')
    
            # Annotate equation
            eqn_text = f'y = {slope:.3f}·x\nR² = {r2:.3f}'
            ax3.text(0.05, 0.95, eqn_text, transform=ax3.transAxes,
                     ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
        ax3.set_title('SWE vs Snow Depth')
        ax3.set_xlabel('Snow Depth (cm)')
        ax3.set_ylabel('SWE (cm)')
        
        # --- Plot-averaged Snow Depth vs SWE with ±1 SD error bars ---
        fig4, ax4 = plt.subplots(figsize=(6.5, 3), constrained_layout=True)  # Snow depth histogram        
        
        # Collapse to plot-level means & SDs
        plot_stats = (
            df[['plot_id', 'snow_depth', 'swe_final']]
            .dropna()
            .groupby('plot_id')
            .agg(
                depth_mean=('snow_depth', 'mean'),
                swe_mean=('swe_final', 'mean'),
                depth_sd=('snow_depth', 'std'),
                swe_sd=('swe_final', 'std')
            )
            .dropna()
        )
        
        # Fit linear regression on plot means
        from sklearn.linear_model import LinearRegression
        X_avg = plot_stats[['depth_mean']].values
        y_avg = plot_stats['swe_mean'].values
        
        reg_avg = LinearRegression(fit_intercept=False)
        reg_avg.fit(X_avg, y_avg)
        slope_avg = float(reg_avg.coef_[0])
        r2_avg = reg_avg.score(X_avg, y_avg)
        
        # Regression line
        x_line_avg = np.linspace(X_avg.min(), X_avg.max(), 100)
        y_line_avg = reg_avg.predict(x_line_avg.reshape(-1, 1))
        ax4.plot(x_line_avg, y_line_avg, color='black', linestyle='--', linewidth=2)
        
        # Error bars and points
        for pid, row in plot_stats.iterrows():
            color = plot_colors.get(pid, 'black')
            ax4.errorbar(
                row['depth_mean'], row['swe_mean'],
                xerr=row['depth_sd'], yerr=row['swe_sd'],
                fmt='o', capsize=3, color=color, ecolor='gray',
                alpha=0.8
            )
            # Add plot_id label next to the point
            ax4.text(
                row['depth_mean'] + 1,  # small horizontal offset to prevent overlap
                row['swe_mean'],
                pid,
                fontsize=8,
                color='black'   ,
                ha='left',
                va='center'
            )

        # Annotations
        eqn_text_avg = f'y = {slope_avg:.3f}·x\nR² = {r2_avg:.3f}'
        ax4.text(0.05, 0.95, eqn_text_avg, transform=ax4.transAxes,
                 ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        # Labels
        ax4.set_title('Plot-Averaged SWE vs Snow Depth ±1 SD')
        ax4.set_xlabel('Mean Snow Depth (cm)')
        ax4.set_ylabel('Mean SWE (cm)')
        
        # Combine the legends
        handles, labels = ax1.get_legend_handles_labels()
        
        # --- Streamlit layout: Map | Legend + Plots ---
        col1, col_group = st.columns([1.8, 3.0], gap="medium")
        
        # --- Map Section ---
        with col1:
            st_folium(m, width=700, height=500, returned_objects=[])
            st.markdown(
                """
                **Note:** A red bounding box should be drawn around all data points. 
                If you do not see the bounding box, there may be erroneous data points 
                plotting outside of the current map view.
                """
            )
        
        # --- Plotting Section (legend + subplots) ---
        with col_group:
            # Top Legend across both plot columns
            fig_legend, ax_legend = plt.subplots(figsize=(8, 1))
            ax_legend.axis('off')
            ax_legend.legend(
                handles, labels,
                loc='center',
                ncol=min(5, len(labels)),  # adjusts based on number of plot IDs
                fontsize='small',
                frameon=False
            )
            st.pyplot(fig_legend)
        
            # 2 sub-columns: histograms and scatterplots
            plot_col1, plot_col2 = st.columns([1.2, 1.3], gap="small")
        
            with plot_col1:
                st.pyplot(fig1)  # Snow depth histogram
                st.pyplot(fig2)  # Density histogram
        
            with plot_col2:
                st.pyplot(fig3)  # SWE vs depth (individual points)
                st.pyplot(fig4)  # SWE vs depth (plot-level means with SD bars)

