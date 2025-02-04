import pandas as pd
import utm

# File path to the CSV
csv_file_path = 'test_data/ACO_form_russellcreekJan2025.csv'

# Define all possible columns as provided in your mapping
all_columns = [
    'start_time', 'study_area', 'users', 'plot_id', 'plot_type', 'sample_type', 'snow_core_number',
    'depth_cm', 'depth_final_cm', 'snow_depth', 'swe_final_swescale', 'depth_max',
    'swe_final_gscale', 'density_gscale', 'density_swescale', 'sample_rating', 'obs_notes',
    'Easting_m', 'Northing_m', 'lat', 'lon'
]

# Initialize an empty DataFrame with all the column names
df = pd.DataFrame(columns=all_columns)

# Define the columns you want to read from the CSV
desired_columns = [
    'Job Start Time',
    'Project Name',
    'Russell Creek Substation',
    'Other Station Name',
    'User ',
    'Snow Course : Add Snow Core : Depth (cm)',
    'Snow Course : Add Snow Core : Depth Max (cm)',
    'Snow Course : Add Snow Core : Depth Max (cm)',
    'Snow Course : Add Snow Core : Depth Max (cm)',
    'Snow Course : Add Snow Core : SWE (cm)',
    'Snow Course : Add Snow Core : Density',
    'Snow Course : Add Snow Core : Core Rating',
    'Snow Course : Add Snow Core : Core Notes',
    'Snow Course : Add Snow Core : Snow Core #'

]

# Read the relevant columns from the CSV
df_data = pd.read_csv('test_data/ACO_wxvisit_surveys_Jan2025.csv', usecols=desired_columns, header=0)

# Apply the condition directly to create the 'plot_id' column
df_data['plot_id'] = df_data.apply(
    lambda row: row['Other Station Name'] if row['Russell Creek Substation'] == 'Other' else row['Russell Creek Substation'], axis=1
)

# Now map the values to the DataFrame 'df' (from the columns you're loading)
df['start_time'] = df_data['Job Start Time']
df['study_area'] = df_data['Project Name']
df['users'] = df_data['User ']
df['depth_cm'] = df_data['Snow Course : Add Snow Core : Depth (cm)']
df['depth_final_cm'] = df_data['Snow Course : Add Snow Core : Depth Max (cm)']
df['snow_depth'] = df_data['Snow Course : Add Snow Core : Depth Max (cm)']
df['swe_final_gscale'] = df_data['Snow Course : Add Snow Core : SWE (cm)']
df['density_gscale'] = df_data['Snow Course : Add Snow Core : Density']
df['sample_rating'] = df_data['Snow Course : Add Snow Core : Core Rating']
df['obs_notes'] = df_data['Snow Course : Add Snow Core : Core Notes']
df['plot_id'] = df_data['plot_id']
df['snow_core_number'] = df_data['Snow Course : Add Snow Core : Snow Core #']
df['plot_type'] = 'snow course'
df['sample_type'] = 'density'

#  drop empty (non-snow course) rows
df = df.dropna(subset=['snow_core_number'])
df = df.dropna(subset=['plot_id'])

# Read the GNSS CSV with plot coordinates
df_gnss = pd.read_csv('test_data/RussellCreek_SnowPlots.csv', usecols=['plot_id', 'Easting_m', 'Northing_m', 'lat', 'lon'])

# Assuming some rows have easting/northing but no lat/lon, or lat/lon but no easting/northing
# We will calculate lat/lon where missing, using UTM Zone 9 (since that's constant)

# Calculate lat/lon for rows where easting/northing are available
df_gnss['lat'], df_gnss['lon'] = zip(*df_gnss.apply(
    lambda row: utm.to_latlon(row['Easting_m'], row['Northing_m'], 9, 'U') if pd.notnull(row['Easting_m']) and pd.notnull(row['Northing_m']) else (row['lat'], row['lon']),
    axis=1
))

# Alternatively, calculate easting/northing for rows with lat/lon available (if needed)
df_gnss['Easting_m'], df_gnss['Northing_m'] = zip(*df_gnss.apply(
    lambda row: utm.from_latlon(row['lat'], row['lon'])[:2] if pd.notnull(row['lat']) and pd.notnull(row['lon']) else (row['Easting_m'], row['Northing_m']),
    axis=1
))

# Loop through the plot_id in df
for index, row in df.iterrows():
    # Get the matching plot_id in df_gnss
    matching_row = df_gnss[df_gnss['plot_id'] == row['plot_id']]

    if not matching_row.empty:
        # If there is a match, update the coordinates in df
        df.at[index, 'Easting_m'] = matching_row['Easting_m'].values[0]
        df.at[index, 'Northing_m'] = matching_row['Northing_m'].values[0]
        df.at[index, 'lat'] = matching_row['lat'].values[0]
        df.at[index, 'lon'] = matching_row['lon'].values[0]

# Now df will have the coordinates from df_gnss where plot_id matches


# Now df will have the coordinates (Easting_m, Northing_m, lat, lon) from df_gnss based on plot_id
# print(df.head())
df = df[all_columns]
df.to_csv('output_file.csv', index=False)