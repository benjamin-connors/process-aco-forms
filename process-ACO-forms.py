import numpy as np
import pandas as pd
import utm
import warnings
import sys
import os
import io
import zipfile
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
import folium
from folium import features
from streamlit_folium import st_folium


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
### App Page Inputs
# study area
study_area = st.selectbox(
    "Select Study Area",
    ('Cruickshank', 'Englishman', 'Metro Vancouver', 'Russell Creek'), index=None)
study_area_dict = {
    'Cruickshank': 'CRU',
    'Englishman': 'ENG',
    'Metro Vancouver': 'MV',
    'Russell Creek': 'TSI'
}
# ACO flight number
ACO_flt_num = st.selectbox('Select ACO Flight/Phase Number', ('01', '02', '03', '04', '05'), index=None)

# .csv files: 1.ACO survey forms 2. GNSS coords
file_dmform = st.file_uploader('**Upload .CSV Containing Device Magic ACO Survey Forms Here**')
file_gnss = st.file_uploader('Upload .CSV Containing GNSS Info Here')

### MAIN PROGRAM
if st.button('Process Forms'):
    # initialize warnings variable
    warn = []
    # get study area abbreviation for output filenames
    study_area_str = study_area_dict[study_area]

    # read dmform file and get column names
    df_dmform = pd.read_csv(file_dmform)
    dmform_cols = df_dmform.columns

    # read gnss file
    df_gnss = pd.read_csv(file_gnss, usecols=['plot_id', 'Easting_m', 'Northing_m'])

    # read in fieldnames spreadsheet (contains current and historical DMform names for both .csv and google sheets, flags for which fields to read, and new column names for output)
    df_fieldnames = pd.read_csv('ACO_form_fieldnames .csv')

    # find fieldnames to keep based on 'read_flag' columns of fielnames file
    ix_keep = df_fieldnames['read_flag'].apply(lambda x: True if x == 1 else False)

    # initialize main dataframe with desired fields
    df = pd.DataFrame(columns=df_fieldnames['post_process'][ix_keep])

    # loop fields that we want to keep
    for ii in df_fieldnames['post_process'][ix_keep].index:
        # if the file contains data for this field (any historical naming convention)
        if any(np.in1d(df_fieldnames.iloc[ii,:], dmform_cols)): 
            # take that data and enter it into new df with post-processing column name
            df[df_fieldnames['post_process'][ii]] = df_dmform[np.unique(df_fieldnames.iloc[ii,np.in1d(df_fieldnames.iloc[ii,:], dmform_cols)])]

    # only keep entries for selected study area
    df = df[df['study_area'] == study_area]
    len_raw = len(df)
           
    #  add ACO flight no.
    df.insert(0, 'aco_flight_number', ACO_flt_num)
  
    # populate snow_depth (== depth values from both density and depth surveys combined in one field)
    df.insert(df.columns.get_loc('multicore'), 'snow_depth',  np.nansum(df['depth_final_cm'] + df['depth_max']))

    # get summary statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df_summary = df.groupby(['aco_flight_number','plot_id']).agg({
        "snow_depth": ["mean", "median", "std", "count"],
        "density_gscale": ["mean", "median", "std", "count"],
        "density_swescale": ["mean", "median", "std", "count"],
        "swe_final_swescale": ["mean", "median", "std", "count"],
        "swe_final_swescale": ["mean", "median", "std", "count"]})

    #  find entries that appear to be cardinal plots, but don't have the plot type entered (give warning and assign cardinal plot type to these entries)
    # WARNING
    if np.any(pd.isnull(df['plot_type']) & ~pd.isnull(df['cardinal_dir'])):
        plot_str = np.unique(df.loc[pd.isnull(df['plot_type']) & ~pd.isnull(df['cardinal_dir']), 'plot_id'])
        warn.append('Input forms for the following plots do not contain ''plot_type'' but appear to contain cardinal plot data: (' +
        ", ".join(plot_str) +
        '). They have been processed as cardinal plots and included in the final output.')
        # change to cardinal
        df.loc[pd.isnull(df['plot_type']) & ~pd.isnull(df['cardinal_dir']), 'plot_type'] = 'Cardinal 10 m'

    # check filter by cardinal plots assign to cardinal variable and to road transect variable.
    if np.sum(df["plot_type"]=="Cardinal 10 m") != len(df):
        # split data into cardinal and not cardinal and add warning
        df_not_cardinal = df[df["plot_type"]!="Cardinal 10 m"]
        warn.append('It appears that some of the input form data is not from cardinal 10 m surveys. These entries have been removed and will be downloaded as XXX.csv')
        df = df[df["plot_type"]=="Cardinal 10 m"]

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
        'SE': 315}

    # get angles for each data point
    ang = df["cardinal_dir"].apply(lambda x: cardinal_ang.get(x))

    # use angles and distance from center to adjust eastings and northings
    ix = ~np.isnan(ang)
    df.loc[ix, 'Easting_m'] = df.loc[ix, 'Easting_m'] + np.round(np.cos(np.deg2rad(ang[ix])), 3) * df.loc[ix,'distance_m']
    df.loc[ix, 'Northing_m'] = df.loc[ix, 'Northing_m'] + np.round(np.sin(np.deg2rad(ang[ix])), 3) * df.loc[ix, 'distance_m']

    # fill empty coordinates
    ix = (df['Easting_m'] < 100) | (df['Northing_m'] < 100) | (pd.isnull(df['Easting_m'])) | (pd.isnull(df['Northing_m']))
    df.loc[ix, 'Easting_m'] = None
    df.loc[ix, 'Northing_m'] = None

    # get lat/lons
    (df['lat'], df['lon']) = utm.to_latlon(df['Easting_m'], df['Northing_m'], 10, 'U')
    df.loc[ix, ['lat', 'lon']] = None

    # DATA/PROCESSING CHECKS
    if np.where(np.isnan(df['distance_m'])):
        warn.append('Distance to center was not provided for all entries. The coordinate fields for these entries have been filled using a fill value of 999,999.')

    # depth values for all entries?
    if np.any(np.isnan(df['snow_depth'])):
        warn.append('Depth values were not found for all entries.')

    # coordinates not calculated for some entries
    if np.any(np.isnan(df['Easting_m'] == None)):
        warn.append('UTM coordinates were not available for all entries. A fill value of 999,999 has been used.')

    # fill any "0" coords (e.g. CRU plot R1O) with 999999 and give warning
    if (np.where(df['Easting_m'] < 100) or np.where(df['Northing_m'] < 100)):
        df.loc[df['Easting_m'] < 100, 'Easting_m'] = None
        df.loc[df['Northing_m'] < 100, 'Northing_m'] = None
        warn.append('Some of the easting/northing coordinates seemed like nonsense and have been replaced with a fill value of 999,999.')

    # at least 1 measurement for each cardinal dir (??)
    stn_str = ""
    for ii in np.unique(df['plot_id']):
        if len(np.unique(df.loc[df['plot_id'] == ii, 'cardinal_dir'])) != 9:
            st_str = stn_str + ii
    if stn_str != "":
        warn.append('The following plots do not contain measurments for each cardinal direction: ' + stn_str)

        # Print Warning Messages
    if warn:
        st.warning('**WARNING:** Not all data was processed successfully!\
            \n Please see warning messages below for details')
        for c, ii in enumerate(warn):
            st.warning('Warning ' + str(c+1) + ': ' + str(ii))
    else:
        st.warning(':green[All data was processed successfully!] 👍')

    # HAS DATA FALLEN THROUGH THE GAPS?
    if (len(df) + len(df_not_cardinal)) != len_raw:
        st.warning(":red[**WARNING: DATA HAS FALLEN THROUGH THE CRACKS!!!**\
             \n Input data has somehow not made it into either the processed or unprocessed output files.\
             \n In other words, some of the data has gone missing.\
                \n Please contact the poor soul responsible for managing this application.\
                \n You can still download the data that was processed in the meantime, but you have been warned!]")

    # zip output files using buffer memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "x") as csv_zip:
        csv_zip.writestr(study_area_str + '_flight' + ACO_flt_num + '_DMform_processed.csv', pd.DataFrame(df).to_csv())
        csv_zip.writestr(study_area_str + '_flight' + ACO_flt_num + '_DMform_NOTprocessed.csv', pd.DataFrame(df_not_cardinal).to_csv())
        csv_zip.writestr(study_area_str + '_flight' + ACO_flt_num + '_DMform_summarystats.csv', pd.DataFrame(df_summary).to_csv())

    # Mapping
    # Remove rows with NaNs in 'lat' or 'lon'
    data_clean = df.dropna(subset=['lat', 'lon'])

    # Create a folium map centered on the average location
    map_center = [data_clean['lat'].mean(), data_clean['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # Add sample locations to the map
    for _, row in data_clean.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['name']
        ).add_to(m)

    # Optionally, add contour data if available
    # contour_geojson = 'path/to/contours.geojson'
    # folium.GeoJson(contour_geojson, name='Contours').add_to(m)

    # Render the map in Streamlit
    st_folium(m, width=700, height=500)



    # # download button
    # st.download_button(
    #     label="Download Output Files",
    #     data=buf.getvalue(),
    #     file_name='ACO_' + study_area_str + '_flight' + ACO_flt_num + '_outputfiles.zip',
    #     mime="application/zip",
    #     )




