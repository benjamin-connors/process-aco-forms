import pandas as pd
import utm

# Assuming your DataFrame is named df
df = pd.read_excel("russellcreek_station_info.xlsx")




# Function to convert Easting/Northing to Lat/Lon
def easting_northing_to_latlon(row):
    if pd.notna(row['Easting_m']) and pd.notna(row['Northing_m']):
        lat, lon = utm.to_latlon(row['Easting_m'], row['Northing_m'], zone_number=10, northern=True)  # Adjust zone_number as needed
        return pd.Series([lat, lon])
    return pd.Series([row['lat'], row['lon']])

# Function to convert Lat/Lon to Easting/Northing
def latlon_to_easting_northing(row):
    if pd.notna(row['lat']) and pd.notna(row['lon']):
        easting, northing, _, _ = utm.from_latlon(row['lat'], row['lon'])
        return pd.Series([easting, northing])
    return pd.Series([row['Easting_m'], row['Northing_m']])

# Apply the conversion functions
df[['lat', 'lon']] = df.apply(easting_northing_to_latlon, axis=1)
df[['Easting_m', 'Northing_m']] = df.apply(latlon_to_easting_northing, axis=1)

df.to_excel('lat_lon_output.xlsx', index=False)