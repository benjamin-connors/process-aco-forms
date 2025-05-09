import pandas as pd
import os
from datetime import datetime

# === USER INPUTS (everything here need to be updated before processing)
# path to fielddata_final (any one is fine, this just copies the column layout)
fielddata_final_path = "S:/ACO/2025/MetroVan/field_data/20250430/processed/20250430_MV_fielddata_final.csv"

# metrovan survey files to process
mv_files = [
    "S:/ACO/2025/MetroVan/field_data/20250430/Snow Survey QC_Palisade_May1-2025.xlsx",
    "S:/ACO/2025/MetroVan/field_data/20250430/Snow Survey QC_Dog_Apr30-2025.xlsx",
    "S:/ACO/2025/MetroVan/field_data/20250430/Snow Survey QC_Grouse_Apr30_2025.xlsx",
    "S:/ACO/2025/MetroVan/field_data/20250430/Snow Survey QC_Orchid_May1-2025.xlsx"
]

# metrovan station names (need to match to order of mv_files)
stn_names = [
    "Palisade Lake",
    "Dog Mountain",
    "Grouse Mountain",
    "Orchid Lake"
]

# survey_id (match to ID of ACO processing)
survey_id = "20250430"
# actual date of metrovan survey (enter one value if same for all files, otherwise match to order of mv_files)
survey_date = datetime.strptime("5/1/2025 12:00", "%m/%d/%Y %H:%M")
# last row of data in survey file (enter one value if same for all files, otherwise match to order of mv_files)
end_row = 50  # Set as needed (can be left long at 50 if a filtering method e.g. INCLUDE_SAMPLE == Yes is being used)

# === MAIN SCRIPT (nothing to edit beyond here)
# get output template headers from fielddata_final file
headers = pd.read_csv(fielddata_final_path, nrows=0).columns.tolist()

# create empty final df
combined_df = pd.DataFrame(columns=headers)

# process each mv survey file
for i, (file2_path, stn_name) in enumerate(zip(mv_files, stn_names)):
    print(f"\nProcessing: {file2_path}")

    # resolve survey_date and end_row (handle single value or list)
    this_survey_date = survey_date[i] if isinstance(survey_date, list) else survey_date
    this_end_row = end_row[i] if isinstance(end_row, list) else end_row

    # create empty df with same columns as final structure
    df = pd.DataFrame(columns=headers)

    # indexing for MV survey file (assumes consistent structure of MV survey files )
    header_row_excel = 12
    data_start_row_excel = 14
    n_data_rows = this_end_row - data_start_row_excel

    # read headers and data
    mv_headers = pd.read_excel(file2_path, sheet_name='template', header=header_row_excel, nrows=1).columns.tolist()
    df_mv = pd.read_excel(file2_path, sheet_name='template', skiprows=data_start_row_excel, nrows=n_data_rows, header=None)
    df_mv.columns = mv_headers

    # filter on 'INCLUDE SAMPLE'
    if 'INCLUDE SAMPLE' in df_mv.columns:
        df_mv = df_mv[df_mv['INCLUDE SAMPLE'].astype(str).str.strip().str.upper() == 'YES']
    else:
        raise KeyError("Column 'INCLUDE SAMPLE' not found in Excel data.")

    # column mapping
    column_mapping = {
        "SD /wo Plug": "snow_depth",
        "SW": "swe_final",
        "SS": "density",
        "Comments": "obs_notes",
    }

    # map data
    for src_col, target_col in column_mapping.items():
        if src_col in df_mv.columns and target_col in df.columns:
            df[target_col] = df_mv[src_col].values

    # add additional info
    df['study_area'] = "Metro Vancouver"
    df['plot_id'] = stn_name
    df['plot_type'] = "snow course"
    df['sample_type'] = "Density"
    df['survey_ID'] = survey_id
    df['start_time'] = this_survey_date

    # append to combined DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# save final output file
output_dir = os.path.dirname(mv_files[0])
processed_dir = os.path.join(output_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)

output_filename = f"{survey_id}_MV_snowcourses_processed.csv"
output_path = os.path.join(processed_dir, output_filename)

combined_df.to_csv(output_path, index=False)
print(f"\nSaved combined data to: {output_path}")
