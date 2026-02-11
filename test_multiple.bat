@echo off
setlocal EnableDelayedExpansion

rem Save start time
set "START=%TIME%"
echo Start: %DATE% %START%

rem ========================


@REM @REM GraphicsLPIPS_FinalNetwork_kfolds
@REM python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM @REM TMQ_NR_1VP_org_kfolds
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_NR_1VP_org_kfolds done
@REM @REM TMQ_NR_4VP_org_kfolds
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_NR_4VP_org_kfolds done
@REM @REM TMQ_OR_1VP_org_kfolds
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_OR_1VP_org_kfolds done
@REM @REM TMQ_OR_4VP_org_kfolds
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_OR_4VP_org_kfolds done
@REM @REM **TSMD_NR_1VP_yf03_kfolds**
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM echo TSMD_NR_1VP_yf03_kfolds done
@REM **TSMD_NR_4VP_yf03_kfolds**
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM echo TSMD_NR_4VP_yf03_kfolds done
@REM @REM **TSMD_NR_8VP_yf03_kfolds**
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM @REM **TMQ_NR_1VP_fib_kfolds**
@REM python config.py -m TMQ_NR_1VP_fib_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_fib_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_fib_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_fib_kfolds -use_folds True -v 1 -vm Fibonacci -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_fib_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"

@REM @REM **TMQ_NR_4VP_fib_kfolds**
@REM python config.py -m TMQ_NR_4VP_fib_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_fib_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_fib_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_fib_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_fib_kfolds -use_folds True -v 4 -vm Fibonacci -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_fib_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_NR_4VP_fib_kfolds -use_folds True -v 4 -vm Fibonacci -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"

@REM **TMQ_NR_8VP_fib_kfolds**
@REM python config.py -m TMQ_NR_8VP_fib_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_8VP_fib_kfolds -use_folds True -v 8 -vm Y_fixed_0.3 -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_8VP_fib_kfolds -use_folds True -v 8 -vm Fibonacci -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_8VP_fib_kfolds -use_folds True -v 8 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TMQ_NR_8VP_fib_kfolds -use_folds True -v 8 -vm Fibonacci -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"

@REM **TMQ_NR_8VP_fib_kfolds**
@REM python config.py -m TMQ_NR_16VP_fib_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_16VP_fib_kfolds -use_folds True -v 16 -vm Fibonacci -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"

@REM @REM **TMQ_NR_1VP_yf03_kfolds**
@REM python config.py -m TMQ_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Fibonacci -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Fibonacci -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"

@REM @REM **TMQ_NR_4VP_yf03_kfolds**
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Fibonacci -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Fibonacci -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"

@REM **TMQ_NR_8VP_fib_kfolds**
@REM python config.py -m TMQ_NR_8VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_8VP_yf03_kfolds -use_folds True -v 8 -vm Y_fixed_0.3 -rm New_Render -db "TMQ" -mos "D:\These\BDD\TexturedMeshes\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_SJTUtest_NR_4VP_yf03_kfolds -v 4 -vm Y_fixed_0.3 -rm New_Render -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_normalized.csv"
@REM python config.py -m TMQ_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_normalized.csv"
@REM @REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 8 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TexturedMeshes\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -v 4 -vm Y_fixed_0 -rm New_Render -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_normalized.csv" --use_folds
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -v 1 -vm Y_fixed_0 -rm 0_0_light -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_normalized.csv" --use_folds
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -v 4 -vm Y_fixed_0 -rm 0_0_light -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_normalized.csv" --use_folds
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -v 8 -vm Y_fixed_0 -rm 0_0_light -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_normalized.csv" --use_folds
@REM python config.py -m SJTU-TMQA_NR_8VP_yf03_kfolds -v 8 -vm Y_fixed_0.3 -rm New_Render -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\folds\SJTU-TMQA_MOS_test20.csv" --use_folds
@REM python config.py -m SJTU-TMQA_NR_16VP_yf03_kfolds -v 16 -vm Y_fixed_0.3 -rm New_Render -db "SJTU-TMQA" -mos ".\dataset\SJTU-TMQA\SJTU-TMQA_MOS_1-5.csv" -testlist ".\dataset\SJTU-TMQA\folds\SJTU-TMQA_MOS_test20.csv" --use_folds

@REM echo TSMD_NR_8VP_yf03_kfolds done
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