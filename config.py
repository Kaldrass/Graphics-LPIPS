import subprocess
import argparse

# model = 'TMQ_NR_1VP_org_kfolds'#'TMQ_OR_1VP_org'#'GraphicsLPIPS_FinalNetwork'#'TMQ_OR_1VP_org' # 'LPIPS' or 'GraphicsLPIPS'
# use_folds = True
# # ----- Testing parameters -----
# testing_views = 4
# view_method = 'Y_fixed_0.3' # 'Original', 'Fibonacci', 'Y_fixed_0.3' or 'Polyhedron'
# render_method = 'New_Render' # 'New_Render' or 'Old_render'
# database = 'TSMD' # 'TSMD' or 'BASICS(PC)_DB' or 'TMQ'
# mos_csv_file = r'D:\These\BDD\TSMD\MOS\TSMD_MOS.csv' if database == 'TSMD' \
#     else r'D:/These/BDD/TMQ/Collected_Data/MOS+CI_3000stimuli.csv' #r'D:/These/BDD/TMQ/Collected_Data/MOS+CI_3000stimuli.csv',s r'D:\These\BDD\TSMD\MOS\TSMD_MOS.csv'#r"D:\These\BDD\BASICS(PC)_DB\MOS_CI.csv" # Depends on the DATABASE used.
# test_list_csv = r'D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%_TestList_scaled.csv' if database == 'TSMD' \
#     else r'D:\These\Graphics-LPIPS\dataset\TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv' #r'D:\These\Graphics-LPIPS\dataset\TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv', r'D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%_TestList_scaled.csv' # We need to take the 1st column of the CSV file as the list of files.

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m','--model', type=str, default="TMQ_NR_1VP_org_kfolds", help='model to evaluate: LPIPS or GraphicsLPIPS')
parser.add_argument('--use_folds', action='store_true', help='use k-folds for testing TMQ or not')
parser.add_argument('-v', '--views', type=int,  help='Number of testing views', default=1)
parser.add_argument('-vm','--view_method', type=str, default="Original", help='view selection method: Original, Fibonacci, Y_fixed_0.3, Polyhedron')
parser.add_argument('-rm','--render_method', type=str, default="New_Render", help='render method: New_Render or Old_render')
parser.add_argument('-db','--database', type=str, default=None, help='database to use: TSMD, BASICS(PC)_DB or TMQ')
parser.add_argument('-mos','--mos_csv_file', type=str, default=None, help='path to the MOS csv file')
parser.add_argument('-testlist','--test_list_csv', type=str, default=None, help='path to the test list csv file')

opt = parser.parse_args() 

model = opt.model
use_folds = opt.use_folds
testing_views = opt.views
view_method = opt.view_method
render_method = opt.render_method
database = opt.database
mos_csv_file = opt.mos_csv_file
test_list_csv = opt.test_list_csv

if __name__ == "__main__":
    # launch Light_GraphicsLPIPS_csv.py and correlation_VP.py after changing these parameters
    cmd = [
    'python', 'Light_GraphicsLPIPS_csv.py',
    '-m', model,
    *(['--use_folds'] if use_folds else []),
    '-v', str(testing_views),
    '-vm', view_method,
    '-rm', render_method,
    '-db', database,
    '-mos', mos_csv_file,
    '-testlist', test_list_csv,
]
    cmd2 = [
           'python', 'correlation_VP.py',
           '-m', model,
           *(['--use_folds'] if use_folds else []),
           '-v', str(testing_views),
           '-vm', view_method,
           '-rm', render_method,
           '-db', database,
    ] 
    subprocess.run(cmd, check=True, text=True)
    subprocess.run(cmd2, check=True, text=True)

