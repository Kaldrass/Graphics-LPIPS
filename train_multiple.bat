@echo off
setlocal EnableDelayedExpansion

rem Save start time
set "START=%TIME%"
echo Start: %DATE% %START%

rem ========================


@REM echo training SJTU-TMQA_NR_NVP_yf03/fib_kfolds...
@REM python train.py --name SJTU-TMQA_NR_4VP_yf0_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Y_fixed_0" --root_refPatches "Source\4VP" --root_distPatches "Distorted\4VP" --datasets "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_train80.csv" --testcsv "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_test20.csv" --target "mos"
@REM python train.py --name SJTU-TMQA_NR_4VP_fib_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Fibonacci" --root_refPatches "Source\4VP" --root_distPatches "Distorted\4VP" --datasets "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_train80.csv" --testcsv "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_test20.csv" --target "mos"
@REM python train.py --name SJTU-TMQA_NR_8VP_yf0_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Y_fixed_0" --root_refPatches "Source\8VP" --root_distPatches "Distorted\8VP" --datasets "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_train80.csv" --testcsv "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_test20.csv" --target "mos"
@REM python train.py --name SJTU-TMQA_NR_8VP_fib_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Fibonacci" --root_refPatches "Source\8VP" --root_distPatches "Distorted\8VP" --datasets "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_train80.csv" --testcsv "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_test20.csv" --target "mos"
@REM @REM python train.py --name SJTU-TMQA_NR_16VP_yf0_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Y_fixed_0" --root_refPatches "Source\16VP" --root_distPatches "Distorted\16VP" --datasets "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_train80.csv" --testcsv "./dataset/SJTU-TMQA/folds/SJTU-TMQA_MOS_test20.csv" --target "mos"
@REM echo training mixed one NO FOLDS...
@REM python train.py --name TMQ-SJTU_NR_4VP_yf --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Y_fixed_0" --different_testset --root_refPatches "Source\4VP" --root_distPatches "Distorted\4VP" --datasets "./dataset/TMQ/folds/TexturedDB_100%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/SJTU-TMQA/SJTU-TMQA_MOS_normalized.csv" --target "mos"
@REM python train.py --name TMQ-SJTU_NR_4VP_fib --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci" "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Fibonacci" --different_testset --root_refPatches "Source\4VP" --root_distPatches "Distorted\4VP" --datasets "./dataset/TMQ/folds/TexturedDB_100%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/SJTU-TMQA/SJTU-TMQA_MOS_normalized.csv" --target "mos"
@REM python train.py --name TMQ-SJTU_NR_8VP_yf --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Y_fixed_0" --different_testset --root_refPatches "Source\8VP" --root_distPatches "Distorted\8VP" --datasets "./dataset/TMQ/folds/TexturedDB_100%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/SJTU-TMQA/SJTU-TMQA_MOS_normalized.csv" --target "mos"
@REM python train.py --name TMQ-SJTU_NR_8VP_fib --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci" "D:\These\Projets\CompareMetrics\out\SJTU-TMQA\New_Render\Fibonacci" --different_testset --root_refPatches "Source\8VP" --root_distPatches "Distorted\8VP" --datasets "./dataset/TMQ/folds/TexturedDB_100%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/SJTU-TMQA/SJTU-TMQA_MOS_normalized.csv" --target "mos"
@REM python train.py --name TMQ_NR_2VP_yf03_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "Source\2VP" --root_distPatches "Distorted\2VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
@REM python train.py --name TMQ_NR_2VP_fib_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci" --root_refPatches "Source\2VP" --root_distPatches "Distorted\2VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
@REM python train.py --name TMQ_NR_3VP_yf03_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "Source\3VP" --root_distPatches "Distorted\3VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
@REM python train.py --name TMQ_NR_1VP_yf03_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "Source\1VP" --root_distPatches "Distorted\1VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
python train.py --name TMQ_NR_1VP_fib_NAA_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci_NAA" --root_refPatches "Source\1VP" --root_distPatches "Distorted\1VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
python train.py --name TMQ_NR_4VP_fib_NAA_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci_NAA" --root_refPatches "Source\4VP" --root_distPatches "Distorted\4VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
python train.py --name TMQ_NR_8VP_fib_NAA_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci_NAA" --root_refPatches "Source\8VP" --root_distPatches "Distorted\8VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"

python train.py --name TMQ_NR_1VP_yf03_NAA_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3_NAA" --root_refPatches "Source\1VP" --root_distPatches "Distorted\1VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
python train.py --name TMQ_NR_4VP_yf03_NAA_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3_NAA" --root_refPatches "Source\4VP" --root_distPatches "Distorted\4VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"
python train.py --name TMQ_NR_8VP_yf03_NAA_kfolds --use_folds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3_NAA" --root_refPatches "Source\8VP" --root_distPatches "Distorted\8VP" --datasets "./dataset/TMQ/folds/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6.csv" --testcsv "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --target "judges"


python config.py -m TMQ_NR_1VP_fib_NAA_kfolds -v 1 -vm Fibonacci_NAA -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --use_folds
python config.py -m TMQ_NR_4VP_fib_NAA_kfolds -v 4 -vm Fibonacci_NAA -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --use_folds
python config.py -m TMQ_NR_8VP_fib_NAA_kfolds -v 8 -vm Fibonacci_NAA -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --use_folds

python config.py -m TMQ_NR_1VP_yf03_NAA_kfolds -v 1 -vm Y_fixed_0.3_NAA -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --use_folds
python config.py -m TMQ_NR_4VP_yf03_NAA_kfolds -v 4 -vm Y_fixed_0.3_NAA -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --use_folds
python config.py -m TMQ_NR_8VP_yf03_NAA_kfolds -v 8 -vm Y_fixed_0.3_NAA -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "./dataset/TMQ/folds/TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv" --use_folds


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

