import subprocess
model = 'TMQ_Gautier_NR_1VPn_Yf'
testing_views = 4
view_method = 'Fibonacci' # 'Fibonacci', 'Y_fixed_0.3' or 'Polyhedron'
render_method = 'New_Render' # 'New_Render' or 'Old_render'
database = 'TMQ' # 'TSMD' or 'BASICS(PC)_DB' or 'TMQ'

if __name__ == "__main__":
    # launch Light_GraphicsLPIPS_csv.py and correlation_VP.py after changing these parameters
    gfxlpips = subprocess.run(['python', 'Light_GraphicsLPIPS_csv.py'], check=True, text=True)
    corrvp = subprocess.run(['python', 'correlation_VP.py'], check=True, text=True)
