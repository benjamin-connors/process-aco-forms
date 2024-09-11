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

### STREAMLIT LAYOUT
st.set_page_config(
    layout="wide"
)

'''
# CHRL ACO Survey Form Processing

1. Select the study area and the ACO flight/phase number for the forms to be processed.
2. In Box 1, upload a .csv file containing the Device Magic ACO survey forms to be processed.
3. In Box 2, upload a .csv containing the GNSS coordinates for each of the plots within the survey.  \n **Important:** GNSS file MUST contain the following headers: 'plot_id', 'Easting_m', 'Northing_m'
4. Click the 'Process Forms' button.
5. Review any warnings and click the "Download Output Files" button.
'''
# Inputs
study_area = st.selectbox("Select Study Area", ('Cruickshank', 'Englishman', 'Metro Vancouver', 'Russell Creek'), index=None)
aco_flt_num = st.selectbox('Select ACO Flight/Phase Number', ('01', '02', '03', '04', '05'), index=None)
file_dmform = st.file_uploader('**Upload .CSV Containing Device Magic ACO Survey Forms Here**')
file_gnss = st.file_uploader('Upload .CSV Containing GNSS Info Here')

study_area_dict = {
    'Cruickshank': 'CRU',
    'Englishman': 'ENG',
    'Metro Vancouver': 'MV',
    'Russell Creek': 'TSI'
}

### FUNCTIONS
def add_notprocessed(index, problem):
    global df, df_notprocessed
    if len(index) != len(df):
        raise ValueError("Boolean index length must match the length of DataFrame 'df'")
    rows_to_add = df[index].copy()
    rows_to_add['problem'] = problem
    df_notprocessed = pd.concat([df_notprocessed, rows_to_add])

### MAIN PROGRAM PROCESSING
if st.button('Process Forms'):
    # check for input files
    if file_dmform is None or file_gnss is None:
        st.warning('Both files must be uploaded to process the forms.')
        st.session_state.data_processed = False
    else:
        # initialize session state variables
        st.session_state.data_processed = False
        st.session_state.warnings = []
        st.session_state.df = None
        st.session_state.df_notprocessed = None
        st.session_state.map = None

       # get other vars 
        study_area_str = study_area_dict[study_area]
        warn_str = '*' + study_area_str + '_flight' + aco_flt_num + '_NOTprocessed.csv*'

        # read dmform file, get column names, get number of rows
        df_dmform = pd.read_csv(file_dmform)
        dmform_cols = df_dmform.columns
        initial_length = len(df_dmform)

        # read gnss file and check for repeat plot_ids
        df_gnss = pd.read_csv(file_gnss, usecols=['plot_id', 'Easting_m', 'Northing_m'])
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
        # Find columns that match the current post_process name
            possible_cols = df_fieldnames.iloc[ii, np.in1d(df_fieldnames.iloc[ii, :], dmform_cols)]
            possible_cols = np.unique(possible_cols[possible_cols.isin(df_dmform.columns)])
            if len(possible_cols) > 0:
                # Check which of these columns contain data
                cols_with_data = [col for col in possible_cols if not df_dmform[col].isnull().all()]
                if len(cols_with_data) == 1:
                    # If only one column contains data, use that column
                    df[df_fieldnames['post_process'][ii]] = df_dmform[cols_with_data[0]]
                elif len(cols_with_data) > 1:
                    # If more than one column contains data, issue a warning
                    warnings.warn(f"Multiple columns contain data for post_process '{df_fieldnames['post_process'][ii]}'. Data from column(s) {cols_with_data} will be used.")
                    # You might decide to handle this case differently, for now, we'll take the first one
                    df[df_fieldnames['post_process'][ii]] = df_dmform[cols_with_data[0]]
                else:
                    # If none of the columns contain data, you may want to handle this case, e.g., by setting NaN or another default value
                    df[df_fieldnames['post_process'][ii]] = np.nan

        df.insert(0, 'aco_flight_number', aco_flt_num)
        df.insert(0, 'row', range(len(df)))
        df_notprocessed = pd.DataFrame(columns=df.columns)
        df_notprocessed.insert(0,'problem', [])

        # only keep entries for selected study area
        if (df['study_area'] != study_area).any():
            ix = df['study_area'] != study_area
            add_notprocessed(ix, 'wrong/no study area')
            st.session_state.warnings.append('Incorrect study area found for some input data. These entries have been added to' + warn_str)
            df = df.loc[~ix]

        # populate snow_depth (== depth values from both density and depth surveys combined in one field)
        snow_depth = np.maximum(df['depth_final_cm'].fillna(-np.inf), df['depth_max'].fillna(-np.inf))
        snow_depth = pd.Series(snow_depth).replace(-np.inf, np.nan)
        df.insert(df.columns.get_loc('multicore'), 'snow_depth', snow_depth)

        # add distance_to_centre for centre plots (sometimes missing)
        df.loc[df['cardinal_dir'] == 'Centre', 'distance_m'] = 0

        # get summary statistics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_summary = df.groupby(['plot_id']).agg({
            "snow_depth": ["mean", "median", "std", "count"],
            "density_gscale": ["mean", "median", "std", "count"],
            "density_swescale": ["mean", "median", "std", "count"],
            "swe_final_swescale": ["mean", "median", "std", "count"],
            "swe_final_swescale": ["mean", "median", "std", "count"]})

        #  find entries that appear to be cardinal plots, but don't have the plot type entered (give warning and assign cardinal plot type to these entries)
        if any(pd.isnull(df['plot_type']) & ~pd.isnull(df['cardinal_dir'])):
            ix = pd.isnull(df['plot_type']) & ~pd.isnull(df['cardinal_dir'])
            # give warning
            plot_str = np.unique(df.loc[ix, 'plot_id'])
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries for the following plots do not contain ''plot_type'' but appear to contain cardinal plot data: (' +
                ", ".join(plot_str) + '). These entries have been processed as cardinal plots and are included in the processed output.')
                        # change to cardinal
            df.loc[ix, 'plot_type'] = 'Cardinal 10 m'

        # keep only cardinal plots, add non-cardinal to not_processed
        if any((df['plot_type'] != 'Cardinal 10 m')):
            ix = df["plot_type"]!="Cardinal 10 m"
            add_notprocessed(ix, 'not a cardinal plot')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries do not appear to be cardinal plots. These entries have been added to ' + warn_str)

        # keep only rows that have coordinates in gnss file
        if any(~np.isin(df['plot_id'], df_gnss['plot_id'])):
            ix = ~np.isin(df['plot_id'], df_gnss['plot_id'])
            plot_str = np.unique(df.loc[ix, 'plot_id'])
            add_notprocessed(ix, 'plot_id not in gnss')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries were not provided coordinates in the GNSS file [' + plot_str + ']. These entries have been added to ' + warn_str)

        # add eastings and northings according to plot id
        df = df.merge(df_gnss, on='plot_id')

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

        # get angles and calc coordinates for each data point
        ang = df["cardinal_dir"].apply(lambda x: cardinal_ang.get(x))
        df['Easting_m'] = df['Easting_m'] + np.round(np.cos(np.deg2rad(ang)), 3) * df['distance_m']
        df['Northing_m'] = df['Northing_m'] + np.round(np.sin(np.deg2rad(ang)), 3) * df['distance_m']

        # remove nonsense/zero-coordinates
        if any((df['Easting_m'] < 100) | (df['Northing_m'] < 100) | (df['Easting_m'].isnull()) | (df['Northing_m'].isnull())):
            ix = (df['Easting_m'] < 100) | (df['Northing_m'] < 100) | (pd.isnull(df['Easting_m'])) | (pd.isnull(df['Northing_m']))
            add_notprocessed(ix, 'bad coordinates')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries were given bad coordinates (check GNSS file). These entries have been added to ' + warn_str)

        # get lat/lons
        (df['lat'], df['lon']) = utm.to_latlon(df['Easting_m'], df['Northing_m'], 10, 'U')

        ### DATA/PROCESSING CHECKS
        #  missing distance to centre
        if any(np.isnan(df['distance_m'])):
            ix = np.isnan(df['distance_m'])
            add_notprocessed(ix, 'missing distance')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries are missing cardinal plot distance. These entries have been added to ' + warn_str)

        # missing snow_depth
        if any(np.isnan(df['snow_depth'])):
            ix = np.isnan(df['snow_depth'])
            add_notprocessed(ix, 'missing snow depth')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries are missing snow depth. These entries have been added to ' + warn_str)

        # # at least 1 measurement for each cardinal dir (??)
        # stn_str = ""
        # for ii in np.unique(df['plot_id']):
        #     if len(np.unique(df.loc[df['plot_id'] == ii, 'cardinal_dir'])) != 9:
        #         st_str = stn_str + ii
        # if stn_str != "":
        #     st.session_state.warnings.append('The following plots do not contain measurments for each cardinal direction: ' + stn_str)

        # update session state variables
        st.session_state.data_processed = True
        st.session_state.df = df
        st.session_state.df_notprocessed = df_notprocessed

    if st.session_state.data_processed:
        # Print Warning Messages
        if st.session_state.warnings == []:
            st.warning(':green[All data was processed successfully!] 👍')
        else:
            st.warning('**WARNING:** Not all entries were processed successfully [' + str(len(df)) + '/' + str(initial_length) + ' processed]. See warning messages below.')

            for ii, warn_str in enumerate(st.session_state.warnings):
                st.warning('Warning ' + str(ii+1) + ': ' + str(warn_str))

        # HAS DATA FALLEN THROUGH THE GAPS?
        if (len(df) + len(df_notprocessed)) != initial_length:
            st.warning(":red[**WARNING: DATA HAS FALLEN INTO THE VOID!**\
                \n There is input data that did not make it into either the processed or not-processed output files.\
                \n In other words, some of the data has gone missing.\
                    \n Please contact the maintainer of this application.\
                    \n You can still download the data that was processed in the meantime, but you have been warned that it is incomplete!]")

        # zip output files using buffer memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "x") as csv_zip:
            csv_zip.writestr(study_area_str + '_flight' + aco_flt_num + '_processed.csv', pd.DataFrame(df).to_csv(index=False))
            csv_zip.writestr(study_area_str + '_flight' + aco_flt_num + '_summarystats.csv', pd.DataFrame(df_summary).to_csv())
            if len(df_notprocessed) > 0:
                # bump index by 2 to align with input form rows
                df_notprocessed.index += 2
                csv_zip.writestr(study_area_str + '_flight' + aco_flt_num + '_NOTprocessed.csv', pd.DataFrame(df_notprocessed).to_csv(index=True, index_label='df_row'))

        # download button
        st.download_button(
            label="Download Output Files",
            data=buf.getvalue(),
            file_name='ACO_' + study_area_str + '_flight' + aco_flt_num + '_outputfiles.zip',
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
            tiles='https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.{ext}',
            name='Stadia Alidade Satellite',
            attr='&copy; CNES, Distribution Airbus DS, © Airbus DS, © PlanetObserver (Contains Copernicus Data) | &copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            ext='jpg'
        ).add_to(m)

        # add a box around sample points
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        folium.Rectangle(bounds=bounds, color='red', fill=False).add_to(m)

        # Calculate mean values for snow_depth and mean of density fields for each plot_id
        mean_values = df.groupby('plot_id').agg({
            'snow_depth': 'mean',
            'density_gscale': 'mean',
            'density_swescale': 'mean'
        }).reset_index()
        mean_values['density'] = mean_values[['density_gscale', 'density_swescale']].mean(axis=1)
        mean_values = mean_values.drop(columns=['density_gscale', 'density_swescale'])

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
        fig1, ax1 = plt.subplots(figsize=(7, 2.5))  # Width = 700px / 100, Height = 250px / 100
        bin_edges = np.histogram_bin_edges(df['snow_depth'].dropna(), bins=30)

        for plot_id, color in zip(unique_plot_ids, color_list):
            subset = df[df['plot_id'] == plot_id]
            hist, _ = np.histogram(subset['snow_depth'].dropna(), bins=bin_edges)
            if plot_id == unique_plot_ids[0]:
                bottom = np.zeros_like(hist)
            else:
                bottom = bottom_stack

            bottom_stack = bottom + hist
            ax1.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', color=color, alpha=0.7, label=f'Plot {plot_id}', bottom=bottom)

        ax1.set_title('Bar Stacked Histogram of Snow Depth')
        ax1.set_xlabel('Snow Depth (cm)')
        ax1.set_ylabel('Count')

        # Density Bar Stacked Plot
        fig2, ax2 = plt.subplots(figsize=(7, 2.5))  # Width = 700px / 100, Height = 250px / 100
        bin_edges_density = np.arange(0, 1.05, 0.05)

        for plot_id, color in zip(unique_plot_ids, color_list):
            subset = df[df['plot_id'] == plot_id]
            hist_density, _ = np.histogram(subset['density'].dropna(), bins=bin_edges_density)
            if plot_id == unique_plot_ids[0]:
                bottom_density = np.zeros_like(hist_density)
            else:
                bottom_density = bottom_density_stack

            bottom_density_stack = bottom_density + hist_density
            ax2.bar(bin_edges_density[:-1], hist_density, width=np.diff(bin_edges_density), edgecolor='black', color=color, alpha=0.7, label=f'Plot {plot_id}', bottom=bottom_density)

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
                **Note:** A red bounding box should be drawn around all data points. If you do not see the bounding box, there may be data points plotting outside of the current map view.
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
        