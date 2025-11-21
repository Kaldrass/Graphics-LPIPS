@REM GraphicsLPIPS_FinalNetwork_kfolds
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
python config.py -m GraphicsLPIPS_FinalNetwork_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"

@REM @REM TMQ_NR_1VP_org_kfolds
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_NR_1VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_NR_1VP_org_kfolds done
@REM @REM TMQ_NR_4VP_org_kfolds
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_NR_4VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_NR_4VP_org_kfolds done
@REM @REM TMQ_OR_1VP_org_kfolds
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_OR_1VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_OR_1VP_org_kfolds done
@REM @REM TMQ_OR_4VP_org_kfolds
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm Old_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM python config.py -m TMQ_OR_4VP_org_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\_TSMD_fulldataset.csv"
@REM echo TMQ_OR_4VP_org_kfolds done
@REM @REM **TSMD_NR_1VP_yf03_kfolds**
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TSMD_NR_1VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM echo TSMD_NR_1VP_yf03_kfolds done
@REM @REM @REM **TSMD_NR_4VP_yf03_kfolds**
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TSMD_NR_4VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM echo TSMD_NR_4VP_yf03_kfolds done
@REM @REM @REM **TSMD_NR_8VP_yf03_kfolds**
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 4 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 1 -vm Original -rm New_Render -db "TMQ" -mos "D:\These\BDD\TMQ\Collected_Data\MOS+CI_3000stimuli.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TexturedDB_20%%_TestList_withnbPatchesPerVP_threth0.6.csv"
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 4 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM python config.py -m TSMD_NR_8VP_yf03_kfolds -use_folds True -v 1 -vm Y_fixed_0.3 -rm New_Render -db "TSMD" -mos "D:\These\BDD\TSMD\MOS\TSMD_MOS.csv" -testlist "D:\These\Graphics-LPIPS\dataset\TSMD\TSMD_20%%_TestList_scaled.csv"
@REM echo TSMD_NR_8VP_yf03_kfolds done
pause
