import geopandas as gpd

# Path to your shapefile (.shp)
shapefile_path = "S:/ACO/plot_locations/2025/SNW_MSS_point.shp"

# Step 1: Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Step 2: Ensure CRS is WGS84 (if not, reproject it)
if gdf.crs is not None and not gdf.crs.to_epsg() == 4326:
    gdf = gdf.to_crs(epsg=4326)

# Step 3: Extract coordinates (longitude, latitude) for point geometries
gdf["longitude"] = gdf.geometry.x
gdf["latitude"] = gdf.geometry.y

# If the geometry is polygons or lines, use the centroid
# gdf["longitude"] = gdf.geometry.centroid.x
# gdf["latitude"] = gdf.geometry.centroid.y

# Step 4: Extract all attribute columns and coordinates into a new DataFrame
df_coords = gdf.drop(columns="geometry")  # Drop geometry if only attributes and coordinates are needed

# Step 5: Inspect result
print(df_coords.head())  # This will show the data with both attributes and coordinates
# df_coords.to_csv("output_with_coordinates_and_attributes.csv", index=False)
