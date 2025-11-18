import subprocess
import argparse

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m','--model', type=str, default=model, help='model to evaluate: LPIPS or GraphicsLPIPS')
parser.add_argument('-use_folds', type=bool, default=use_folds, help='use k-folds for testing TMQ or not')
parser.add_argument('-v', '-views', type=int,  help='Number of testing views', default=testing_views)
parser.add_argument('-vm','--view_method', type=str, default=view_method, help='view selection method: Original, Fibonacci, Y_fixed_0.3, Polyhedron')
parser.add_argument('-rm','--render_method', type=str, default=render_method, help='render method: New_Render or Old_render')
parser.add_argument('-db','--database', type=str, default=database, help='database to use: TSMD, BASICS(PC)_DB or TMQ')
parser.add_argument('-mos','--mos_csv_file', type=str, default=mos_csv_file, help='path to the MOS csv file')
parser.add_argument('-testlist','--test_list_csv', type=str, default=test_list_csv, help='path to the test list csv file')

opt = parser.parse_args() 


if __name__ == "__main__":
    
    model = opt.model
    use_folds = opt.use_folds
    testing_views = opt.v
    view_method = opt.view_method
    render_method = opt.render_method
    database = opt.database
    mos_csv_file = opt.mos_csv_file
    test_list_csv = opt.test_list_csv
    
    # launch Light_GraphicsLPIPS_csv.py and correlation_VP.py after changing these parameters
    gfxlpips = subprocess.run(['python', 'Light_GraphicsLPIPS_csv.py'], check=True, text=True)
    # corrvp = subprocess.run(['python', 'correlation_VP.py'], check=True, text=True)

