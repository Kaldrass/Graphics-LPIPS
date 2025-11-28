@echo off
setlocal EnableDelayedExpansion

rem Save start time
set "START=%TIME%"
echo Start: %DATE% %START%

rem ========================

@REM python train.py --name TMQ_NR_1VP_org_kfolds_new --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Original" --root_refPatches "\Source\1VP" --root_distPatches "\Distorted\1VP" --target "judges"
@REM echo training TMQ_NR_1VP_org_kfolds_new done.
@REM python train.py --name TMQ_NR_4VP_org_kfolds_new --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Original" --root_refPatches "\Source\4VP" --root_distPatches "\Distorted\4VP" --target "judges"
@REM echo training TMQ_NR_4VP_org_kfolds_new done.
@REM python train.py --name TMQ_OR_1VP_org_kfolds_new --src_root "D:\These\Projets\CompareMetrics\out\TMQ\Old_Render\Original" --root_refPatches "\Source\1VP" --root_distPatches "\Distorted\1VP" --target "judges"
@REM echo training TMQ_OR_1VP_org_kfolds_new done.
@REM python train.py --name TMQ_OR_4VP_org_kfolds_new --src_root "D:\These\Projets\CompareMetrics\out\TMQ\Old_Render\Original" --root_refPatches "\Source\4VP" --root_distPatches "\Distorted\4VP" --target "judges"
@REM echo training TMQ_OR_4VP_org_kfolds_new done.
@REM --------------------------------- Those are not rendered with TAA yet.
@REM python train.py --name TMQ_NR_1VP_fib_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci" --root_refPatches "\Source\1VP" --root_distPatches "\Distorted\1VP" --target "judges"
@REM echo training TMQ_NR_1VP_fib_kfolds done.
@REM python train.py --name TMQ_NR_4VP_fib_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci" --root_refPatches "\Source\4VP" --root_distPatches "\Distorted\4VP" --target "judges"
@REM echo training TMQ_NR_4VP_fib_kfolds done.
@REM python train.py --name TMQ_NR_8VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "\Source\8VP" --root_distPatches "\Distorted\8VP" --target "judges"
@REM echo training TMQ_NR_8VP_yf03_kfolds done.
python train.py --name TMQ_NR_16VP_fib_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci" --root_refPatches "\Source\16VP" --root_distPatches "\Distorted\16VP" --target "judges"
echo training TMQ_NR_16VP_fib_kfolds done.
@REM ---------------------------------
@REM python train.py --name TMQ_NR_1VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "\Source\1VP" --root_distPatches "\Distorted\1VP" --target "judges"
@REM echo training TMQ_NR_1VP_yf03_kfolds done.
@REM python train.py --name TMQ_NR_4VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "\Source\4VP" --root_distPatches "\Distorted\4VP" --target "judges"
@REM echo training TMQ_NR_4VP_yf03_kfolds done.
rem ========================

timeout /t 3 >nul

rem Save end time
set "END=%TIME%"
echo End  : %DATE% %END%

rem ----- Parse START (HH:MM:SS,CC) -----
for /f "tokens=1-4 delims=:.," %%A in ("%START%") do (
    set /a SH=1%%A-100
    set /a SM=1%%B-100
    set /a SS=1%%C-100
    set /a SC=1%%D-100
)

rem ----- Parse END (HH:MM:SS,CC) -----
for /f "tokens=1-4 delims=:.," %%A in ("%END%") do (
    set /a EH=1%%A-100
    set /a EM=1%%B-100
    set /a ES=1%%C-100
    set /a EC=1%%D-100
)

rem Convert start and end to centiseconds
set /a START_CS=((SH*60+SM)*60+SS)*100+SC
set /a END_CS=((EH*60+EM)*60+ES)*100+EC

rem Handle midnight wrap
if !END_CS! LSS !START_CS! set /a END_CS+=24*60*60*100

rem Duration in centiseconds
set /a DIFF_CS=END_CS-START_CS

rem Convert back to H M S CS
set /a DH=DIFF_CS/(3600*100)
set /a DIFF_CS%%=3600*100

set /a DM=DIFF_CS/(60*100)
set /a DIFF_CS%%=60*100

set /a DS=DIFF_CS/100
set /a DC=DIFF_CS%%100

echo Duration: !DH!h !DM!m !DS!s !DC!cs

endlocal
pause

