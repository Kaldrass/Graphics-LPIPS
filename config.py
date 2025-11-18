import subprocess

model = 'TMQ_NR_1VP_org_kfolds'#'TMQ_OR_1VP_org'#'GraphicsLPIPS_FinalNetwork'#'TMQ_OR_1VP_org' # 'LPIPS' or 'GraphicsLPIPS'
use_folds = True
# ----- Testing parameters -----
testing_views = 4
view_method = 'Y_fixed_0.3' # 'Original', 'Fibonacci', 'Y_fixed_0.3' or 'Polyhedron'
render_method = 'New_Render' # 'New_Render' or 'Old_render'
database = 'TSMD' # 'TSMD' or 'BASICS(PC)_DB' or 'TMQ'
mos_csv_file = r'D:\These\BDD\TSMD\MOS\TSMD_MOS.csv' if database == 'TSMD' \
    else r'D:/These/BDD/TMQ/Collected_Data/MOS+CI_3000stimuli.csv' #r'D:/These/BDD/TMQ/Collected_Data/MOS+CI_3000stimuli.csv',s r'D:\These\BDD\TSMD\MOS\TSMD_MOS.csv'#r"D:\These\BDD\BASICS(PC)_DB\MOS_CI.csv" # Depends on the DATABASE used.
test_list_csv = r'D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%_TestList_scaled.csv' if database == 'TSMD' \
    else r'D:\These\Graphics-LPIPS\dataset\TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv' #r'D:\These\Graphics-LPIPS\dataset\TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv', r'D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%_TestList_scaled.csv' # We need to take the 1st column of the CSV file as the list of files.

if __name__ == "__main__":
    # launch Light_GraphicsLPIPS_csv.py and correlation_VP.py after changing these parameters
    gfxlpips = subprocess.run(['python', 'Light_GraphicsLPIPS_csv.py'], check=True, text=True)
    corrvp = subprocess.run(['python', 'correlation_VP.py'], check=True, text=True)

