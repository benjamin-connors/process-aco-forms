import pandas as pd
import utm
import streamlit as st
import numpy as np
import warnings
import zipfile
import io

### STREAMLIT LAYOUT
st.set_page_config(
    layout="wide"
)

'''
# CHRL WX Station Visit  Form Processing

1. In Box 1, upload a file (.csv or .xlsx) containing the Device Magic Wx Station Visit forms to be processed.
2. In Box 2, upload a file (.csv or .xlsx) containing the GNSS coordinates for each of the plots within the survey.  \n **Important:** GNSS file MUST contain the following headers: 'plot_id', 'easting_m', 'northing_m'
3. Select the study area and the survey date for the forms to be processed.
4. Click the 'Process Forms' button.
5. Review any warnings and click the "Download Output Files" button.
'''
# Inputs
wxform_file = st.file_uploader('**Upload File Containing Device Magic Wx Station Visit Forms Here**')
gnss_file = st.file_uploader('Upload File Containing GNSS Info Here')
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

# Define the columns you want to read from the CSV
wxform_cols2read = [
    'Job Start Time',
    'Project Name',
    'Russell Creek Substation',
    'Other Station Name',
    'User ',
    'Snow Course : Add Snow Core : Depth Max (cm)',
    'Snow Course : Add Snow Core : SWE (cm)',
    'Snow Course : Add Snow Core : Density',
    'Snow Course : Add Snow Core : Core Rating',
    'Snow Course : Snow Course Notes',
    'Snow Course : Add Snow Core : Snow Core #'
]

output_cols = [
    "survey_ID", "start_time", "study_area", "other_study_area", "users", 
    "plot_id", "pre_survey_notes", "plot_type", "cardinal_dir", "other_direction", 
    "distance_m", "custom_distance_m", "sample_type", "scale_type", "multicore", 
    "plot_features", "snow_depth", "swe_final", "density",
    "sample_rating", "obs_notes", "easting_m", "northing_m", "lat", "lon", "flag"
]

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
    if wxform_file is None or gnss_file is None:
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
        dmform_filename = wxform_file.name.lower()
        gnss_filename = gnss_file.name.lower()

        # Check if the file is CSV (uploaded via Streamlit)
        dmform_filename = wxform_file.name.lower()
    
        if dmform_filename.endswith(".csv"):
            # Read the first line to check if it contains a separator declaration
            first_line = wxform_file.getvalue().decode('utf-8').splitlines()[0]
            
            # Check if the first line contains a separator declaration
            if first_line.startswith('sep='):
                separator = first_line.split('=')[1].strip()  # Extract the separator
                # Rewind the file to the start (after reading the first line)
                wxform_file.seek(0)
                df_wxform = pd.read_csv(wxform_file, sep=separator, usecols=wxform_cols2read, header=1)  # Skip first line
            else:
                separator = ','  # Default separator
                # Rewind the file to the start (if no separator declaration)
                wxform_file.seek(0)
                df_wxform = pd.read_csv(wxform_file, sep=separator, usecols=wxform_cols2read, header=0)  # Read normally
    
        elif dmform_filename.endswith(".xlsx"):
            # Handle Excel files if needed
            df_wxform = pd.read_excel(wxform_file, usecols=wxform_cols2read, header=0)
        
        # Get the column names and number of rows for reference
        wxform_cols = df_wxform.columns
        initial_length = len(df_wxform)
        
        # read gnss file
        if gnss_filename.endswith(".csv"):
            df_gnss = pd.read_csv(gnss_file, usecols=['plot_id', 'easting_m', 'northing_m'])
        elif gnss_filename.endswith(".xlsx"):
            df_gnss = pd.read_excel(gnss_file, usecols=['plot_id', 'easting_m', 'northing_m'])

        # # Remove rows with empty plot_ids
        # df_gnss = df_gnss.dropna(subset=['plot_id'])

        if df_gnss['plot_id'].duplicated().any():
            st.error('**ERROR:** The provided GNSS file contains multiple entries for at least one plot ID. Please update the GNSS file so that it contains at most 1 set of coordinates for each plot ID.')
            st.stop()
            
        # Initialize an empty DataFrame with all the column names
        # df = pd.DataFrame(columns=output_cols)     
        df = pd.DataFrame()
        # Apply the condition directly to create the 'plot_id' column
        df_wxform['plot_id'] = df_wxform.apply(
            lambda row: row['Other Station Name'] if row['Russell Creek Substation'] == 'Other' else row['Russell Creek Substation'], axis=1
        )
                
        # Now map the values to the DataFrame 'df' (from the columns you're loading)
        df['start_time'] = df_wxform['Job Start Time']
        df['study_area'] = df_wxform['Project Name']
        df['users'] = df_wxform['User ']
        df['snow_depth'] = df_wxform['Snow Course : Add Snow Core : Depth Max (cm)']
        df['swe_final'] = df_wxform['Snow Course : Add Snow Core : SWE (cm)']
        df['density'] = df_wxform['Snow Course : Add Snow Core : Density']
        df['sample_rating'] = df_wxform['Snow Course : Add Snow Core : Core Rating']
        df['obs_notes'] = df_wxform['Snow Course : Snow Course Notes']
        df['plot_id'] = df_wxform['plot_id']
        df['plot_type'] = 'Snow Course'
        df['sample_type'] = 'Density'
        
        # set scale type to mass scale for density measurements
        df['scale_type'] = None
        df.loc[df['sample_type'] == 'Density', 'scale_type'] = 'Mass'

        # get survey_date (first date of survey data)generate warning string and add survey_ID
        study_area_str = study_area_dict[study_area]
        survey_date_str = survey_date.strftime("%Y%m%d")
        warn_str = '*' + survey_date_str + '_' + study_area_str+ '_snowcourse_NOTprocessed.csv*'
        df['survey_ID'] = survey_date_str
        
        # Read the gnss file and m and merge easting and northing on plot_id
        df_gnss = pd.read_excel('S:/ACO/plot_locations/2025/2025_TSI_snowplot_locations.xlsx', usecols=['plot_id', 'easting_m', 'northing_m'])
        df = df.merge(df_gnss, on='plot_id', how='left')
        (df['lat'], df['lon']) = utm.to_latlon(df['easting_m'], df['northing_m'], utm_dict[study_area], 'U')
        
        # create not processed df
        df_notprocessed = pd.DataFrame(columns=df.columns)
        df_notprocessed.insert(0,'problem', [])
        
        # only keep entries for selected study area
        if (df['study_area'] != study_area).any():
            ix = df['study_area'] != study_area
            add_notprocessed(ix, 'wrong/no study area')
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries do not have the correct study area. These entries have been added to' + warn_str)
            df = df.loc[~ix]
            
        # only keep entries that have a plot id
        if df['plot_id'].isna().any():            
            ix = df['plot_id'].isna()
            add_notprocessed(ix, 'missing plot_id')
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries are missing a plot_id. These entries have been added to' + warn_str)
            df = df.loc[~ix]
            
        # only keep entries with density data
        if df['density'].isna().any():            
            ix = df['density'].isna()
            add_notprocessed(ix, 'no density data')
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries do not have density data. These entries have been added to' + warn_str)
            df = df.loc[~ix]
            
        # keep only rows that have coordinates in gnss file
        if any(~np.isin(df['plot_id'], df_gnss['plot_id'])):
            ix = ~np.isin(df['plot_id'], df_gnss['plot_id'])
            plot_str = np.unique(df.loc[ix, 'plot_id'])
            add_notprocessed(ix, 'plot_id not in gnss')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries were not provided coordinates in the GNSS file [' +
                ", ".join(plot_str) + ']. These entries have been added to ' + warn_str)
            
        # remove nonsense/zero-coordinates
        if any((df['easting_m'] < 100) | (df['northing_m'] < 100) | (df['easting_m'].isnull()) | (df['northing_m'].isnull())):
            ix = (df['easting_m'] < 100) | (df['northing_m'] < 100) | (pd.isnull(df['easting_m'])) | (pd.isnull(df['northing_m']))
            add_notprocessed(ix, 'bad coordinates')
            df = df.loc[~ix]
            st.session_state.warnings.append('Some [' + str(sum(ix)) + '/' + str(initial_length) + '] entries were given bad coordinates (check GNSS file). These entries have been added to ' + warn_str)

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
        # Keep only desired columns
        df = df[output_cols]
        
        # add flags
        ix = df['sample_rating'] <= 3      
        df.loc[ix, 'flag'] = 'QUALITY'

        # zip output files using buffer memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "x") as csv_zip:
            csv_zip.writestr(survey_date_str + '_' + study_area_str + '_snowcourse_processed.csv', pd.DataFrame(df).to_csv(index=False))
            csv_zip.writestr(survey_date_str + '_' +  study_area_str + '_snowcourse_summarystats.csv', pd.DataFrame(df_summary).to_csv(index=False))
            if len(df_notprocessed) > 0:
                # bump index by 2 to align with input form rows
                df_notprocessed.index += 2
                csv_zip.writestr(survey_date_str + '_' +  study_area_str + '_snowcourse_NOTprocessed.csv', pd.DataFrame(df_notprocessed).to_csv(index=True, index_label='input_row'))

        # download button
        st.download_button(
            label="Download Output Files",
            data=buf.getvalue(),
            file_name=survey_date_str + '_' +  study_area_str + '_snowcourse_outputfiles.zip',
            mime="application/zip",
            )
