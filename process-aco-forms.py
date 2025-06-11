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
# from processing_utils import add_to_df_notprocessed

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
dmform_file = st.file_uploader('**Upload .CSV Containing Device Magic ACO Survey Forms Here**')
gnss_file = st.file_uploader('Upload .CSV Containing GNSS Info Here')
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

### FUNCTIONS
def add_to_df_notprocessed(index, problem):
    global df, df_notprocessed
    if len(index) != len(df):
        raise ValueError("Boolean index length must match the length of DataFrame 'df'")
    rows_to_add = df[index].copy()
    rows_to_add['problem'] = problem
    df_notprocessed = pd.concat([df_notprocessed, rows_to_add])

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
            "sample_rating", "obs_notes", "easting_m", "northing_m", "lat", "lon", "flag"
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
            df_gnss = pd.read_csv(gnss_file, usecols=['plot_id', 'easting_m', 'northing_m'])
        elif gnss_filename.endswith(".xlsx"):
            df_gnss = pd.read_excel(gnss_file, usecols=['plot_id', 'easting_m', 'northing_m'])

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
        id_cols = ['plot_id', 'plot_id_tsi', 'plot_id_cru', 'plot_id_eng', 'plot_id_mv']
        non_null_counts = df[id_cols].notna().sum(axis=1)
        if (non_null_counts > 1).any():
            raise ValueError("Multiple plot_id values found in the same row.")
        fallback_cols = ['plot_id_tsi', 'plot_id_cru', 'plot_id_eng', 'plot_id_mv']
        df['plot_id'] = df[['plot_id'] + fallback_cols].bfill(axis=1).iloc[:, 0]
        df.drop(columns=fallback_cols, inplace=True)

        # enforce formatting
        df['distance_m'] = pd.to_numeric(df['distance_m'], errors='coerce')
        
        # generate 'zero' measurements for 'no snow' plots
        # Identify plots marked as "no snow"
        no_snow_ix = df["is_there_snow"].str.lower() == "no"
        
        # remove and flag any no-snow entries that have real data
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
                    
        # add survey date col to df 
        df.insert(0, 'survey_ID', survey_date_str)
        # create not processed df
        df_notprocessed = pd.DataFrame(columns=df.columns)
        df_notprocessed.insert(0,'problem', [])

        # only keep entries for selected study area
        if (df['study_area'] != study_area).any():
            ix = df['study_area'] != study_area
            add_to_df_notprocessed(ix, 'wrong/no study area')
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries have the wrong/no study area. These entries have been added to' + warn_str)
            df = df.loc[~ix]
            
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
        df['swe_final'] = np.nan
        df['density'] = np.nan
        df['scale_type'] = np.nan

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
            add_to_df_notprocessed(ix, 'contains both mass-scale and swe-scale measurements')
            df = df.loc[~ix]
            st.session_state.warnings.append(
                'Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries contain both mass-scale and swe-scale measurements. '
                'These entries have been added to ' + warn_str)
            
        # add distance_to_centre for centre plots (sometimes missing)
        df.loc[df['cardinal_dir'] == 'Centre', 'distance_m'] = 0

        # Keep only rows that have both a direction (cardinal_dir or other_direction) and a distance (distance_m or custom_distance_m)
        if any(pd.isnull(df['cardinal_dir']) & pd.isnull(df['other_direction']) | 
            pd.isnull(df['distance_m']) & pd.isnull(df['custom_distance_m'])):
            
            ix = (pd.isnull(df['cardinal_dir']) & pd.isnull(df['other_direction'])) | \
                (pd.isnull(df['distance_m']) & pd.isnull(df['custom_distance_m']))
                            
            plot_str = np.unique(df.loc[ix, 'plot_id'])
            add_to_df_notprocessed(ix, 'missing direction or distance data')
            df = df.loc[~ix]
            
            st.session_state.warnings.append(
                'Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries do not contain both required direction and distance data. '
                'These entries have been added to ' + warn_str + ' for the following plots: (' + ", ".join(plot_str) + ').'
            )

        # keep only rows that have coordinates in gnss file
        if any(~np.isin(df['plot_id'], df_gnss['plot_id'])):
            ix = ~np.isin(df['plot_id'], df_gnss['plot_id'])
            plot_str = np.unique(df.loc[ix, 'plot_id'])
            add_to_df_notprocessed(ix, 'plot_id not in gnss')
            df = df.loc[~ix]

            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries were not provided coordinates in the GNSS file [' +
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
            add_to_df_notprocessed(ix, 'missing distance')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries are missing cardinal plot distance. These entries have been added to ' + warn_str)

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
            add_to_df_notprocessed(ix, 'bad coordinates')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries were given bad coordinates (check GNSS file). These entries have been added to ' + warn_str)

        # get lat/lons
        (df['lat'], df['lon']) = utm.to_latlon(df['easting_m'], df['northing_m'], utm_dict[study_area], 'U')

        ### DATA/PROCESSING CHECKS
        # missing snow_depth
        if any(np.isnan(df['snow_depth'])):
            ix = np.isnan(df['snow_depth'])
            add_to_df_notprocessed(ix, 'missing snow depth')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries are missing snow depth. These entries have been added to ' + warn_str)

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
        if (len(df) + len(df_notprocessed)) != initial_length:
            st.warning(":red[**WARNING: DATA HAS FALLEN INTO THE VOID!**\
                \n There is input data that did not make it into either the processed or not-processed output files.\
                \n In other words, some of the data has gone missing.\
                    \n You can still download the data that was processed, but you have been warned that it is incomplete!]")
                       
        # get summary statistics
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
            
        # Add missing columns (empty)
        for col in output_cols:
            if col not in df.columns:
                df[col] = None
        
        # Add flags with appending
        df.loc[~df['sample_rating'].isin(['Good', 'Excellent']), 'flag'] = df['flag'].fillna('') + 'QUALITY'
        df.loc[df['multicore'] == 'yes', 'flag'] = df.apply(
            lambda row: row['flag'] + ', MULTI' if pd.notna(row['flag']) else 'MULTI', axis=1)
        
        df.loc[df['retrieval'] < 60, 'flag'] = df.apply(
            lambda row: row['flag'] + ', RETRIEVAL < 60' if pd.notna(row['flag']) else 'RETRIEVAL < 60', axis=1)
        df.loc[df['retrieval'] > 90, 'flag'] = df.apply(
            lambda row: row['flag'] + ', RETRIEVAL > 90' if pd.notna(row['flag']) else 'RETRIEVAL > 90', axis=1)

        
        # Keep only desired columns
        df = df[output_cols]

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

        # add a tile layer
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            name='Stadia Alidade Smooth',
            attr='&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a>, '
                '&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a>, '
                '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            ext='png'
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

        # Get colormap for each plot id
        plot_colors = pc.qualitative.Light24[:len(unique_combinations['plot_id'].unique())]
        plot_colors = dict(zip(unique_combinations['plot_id'].unique(), plot_colors))

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
        fig1, ax1 = plt.subplots(figsize=(7, 2.5))
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
        fig2, ax2 = plt.subplots(figsize=(7, 2.5))
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
        
        # Combine the legends
        handles, labels = ax1.get_legend_handles_labels()

        # Streamlit columns layout
        col1, col2, col3 = st.columns([2, 1.5, 1], vertical_alignment='top')  # Adjust the column widths as needed

        with col1:

            st_folium(m, width=700, height=500, returned_objects=[])
            st.markdown(
                """
                **Note:** A red bounding box should be drawn around all data points. If you do not see the bounding box, there may be erroneous data points plotting outside of the current map view.
                """
            )

        with col2:
            # Plot the histograms
            st.pyplot(fig1)
            st.pyplot(fig2)
        with col3:
            # Create a new figure for the legend
            fig_legend, ax_legend = plt.subplots(figsize=(3, 5))  # Adjust size as needed
            ax_legend.axis('off')  # Turn off axis
            ax_legend.legend(handles, labels, loc='upper left', fontsize='small') #, bbox_to_anchor=(1, 1), ncol=1)  # Position legend
            st.pyplot(fig_legend)
        
