import csv
import os

## This script checks if there are any folders in the specified directory that do not match the object names in the CSV file.
## It prints the names of any extra folders found.

# CSV
csv_path = r"D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv"
# Folder path containing 3000+ folders
folders_path = r"D:\These\Projets\CompareMetrics\out\TMQ_dis_fib_20VP_650x550_bis"

# Reading CSV file to get object names
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip header
    object_names = set(row[0] for row in reader)

# Listing all folders in the specified directory
folder_names = set(os.listdir(folders_path))

# Comparing folder names with object names
extra_folders = folder_names - object_names

if extra_folders:
    print("Extra folders found :")
    for folder in extra_folders:
        print(folder)
else:
    print("No extra folder found !")
