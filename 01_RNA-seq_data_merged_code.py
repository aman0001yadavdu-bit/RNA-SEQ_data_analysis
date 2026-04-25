# Install dependencies
!pip install pandas openpyxl xlrd

import pandas as pd
from google.colab import files
import os

# Safe file reader
def read_file(file_name, sheet_name=None):
    ext = os.path.splitext(file_name)[1].lower()

    if ext == '.xlsx':
        return pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl')
    elif ext == '.xls':
        return pd.read_excel(file_name, sheet_name=sheet_name, engine='xlrd')
    elif ext == '.csv':
        return pd.read_csv(file_name)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Upload files
print("Upload FIRST file (norm):")
uploaded1 = files.upload()

print("\nUpload SECOND file:")
uploaded2 = files.upload()

file1_name = list(uploaded1.keys())[0]
file2_name = list(uploaded2.keys())[0]

# -------- FIRST FILE --------
df1 = read_file(file1_name)

# 🔥 Fix: if dict (multiple sheets), take first sheet
if isinstance(df1, dict):
    print("First file had multiple sheets, using first sheet automatically.")
    df1 = list(df1.values())[0]

if 'Gene_id' not in df1.columns:
    raise ValueError("First file must contain 'Gene_id' column")

# Add prefix safely
df1['Gene_id'] = df1['Gene_id'].astype(str).apply(
    lambda x: x if x.startswith("Gene_") else f"Gene_{x}"
)

print("\nFirst file preview:")
print(df1.head())

# -------- SECOND FILE --------
ext2 = os.path.splitext(file2_name)[1].lower()

if ext2 in ['.xlsx', '.xls']:
    xls = pd.ExcelFile(file2_name)

    print("\nAvailable sheets:")
    for i, sheet in enumerate(xls.sheet_names):
        print(f"{i}: {sheet}")

    sheet_index = int(input("\nEnter sheet number: "))
    sheet_name = xls.sheet_names[sheet_index]

    df2 = read_file(file2_name, sheet_name=sheet_name)

else:
    df2 = read_file(file2_name)

# 🔥 Fix again (just in case)
if isinstance(df2, dict):
    print("Second file returned multiple sheets, using first one.")
    df2 = list(df2.values())[0]

if 'Gene_id' not in df2.columns:
    raise ValueError("Second file must contain 'Gene_id' column")

print("\nSecond file preview:")
print(df2.head())

# -------- MERGE --------
merged_df = pd.merge(df1, df2, on='Gene_id', how='outer')

print("\nMerged preview:")
print(merged_df.head())

# -------- REMOVE NaN --------
cleaned_df = merged_df.dropna()

print("\nAfter removing NaN rows:")
print(cleaned_df.head())

# -------- SAVE --------
output_file = "merged_cleaned_output.xlsx"
cleaned_df.to_excel(output_file, index=False)

files.download(output_file)

print("\n✅ Done! File saved as:", output_file)
