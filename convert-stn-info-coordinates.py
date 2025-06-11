import pandas as pd
import utm

# Assuming your DataFrame is named df
df = pd.read_excel("S:/ACO/plot_locations/2025/2025_TSI_wxstation_locations.xlsx")

# Function to convert Easting/Northing to Lat/Lon
def easting_northing_to_latlon(row):
    if pd.notna(row['easting_m']) and pd.notna(row['northing_m']):
        lat, lon = utm.to_latlon(row['easting_m'], row['northing_m'], zone_number=9, northern=True)  # Adjust zone_number as needed
        return pd.Series([lat, lon])
    return pd.Series([row['latitude'], row['longitude']])

# Function to convert Lat/Lon to Easting/Northing
def latlon_to_easting_northing(row):
    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
        easting, northing, _, _ = utm.from_latlon(row['latitude'], row['longitude'])
        return pd.Series([easting, northing])
    return pd.Series([row['easting_m'], row['northing_m']])

# Apply the conversion functions
df[['latitude', 'longitude']] = df.apply(easting_northing_to_latlon, axis=1)
df[['easting_m', 'northing_m']] = df.apply(latlon_to_easting_northing, axis=1)

df.to_excel('lat_lon_output.xlsx', index=False)