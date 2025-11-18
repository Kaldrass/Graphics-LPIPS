python train.py --name TSMD_NR_1VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TSMD\New_Render\Y_fixed_0.3" --root_refPatches "\Source\1VP" --root_distPatches "\Distorted\1VP" --target "mos"
echo training TSMD_NR_1VP_yf03_kfolds done.
python train.py --name TSMD_NR_4VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TSMD\New_Render\Y_fixed_0.3" --root_refPatches "\Source\4VP" --root_distPatches "\Distorted\4VP" --target "mos"
echo training TSMD_NR_4VP_yf03_kfolds done.
python train.py --name TSMD_NR_8VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TSMD\New_Render\Y_fixed_0.3" --root_refPatches "\Source\8VP" --root_distPatches "\Distorted\8VP" --target "mos"
echo training TSMD_NR_8VP_yf03_kfolds done.
pause