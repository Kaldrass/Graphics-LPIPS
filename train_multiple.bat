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
python train.py --name TMQ_NR_8VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "\Source\8VP" --root_distPatches "\Distorted\8VP" --target "judges"
echo training TMQ_NR_8VP_yf03_kfolds done.
python train.py --name TMQ_NR_16VP_fib_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Fibonacci" --root_refPatches "\Source\16VP" --root_distPatches "\Distorted\16VP" --target "judges"
echo training TMQ_NR_16VP_fib_kfolds done.
@REM ---------------------------------
@REM python train.py --name TMQ_NR_1VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "\Source\1VP" --root_distPatches "\Distorted\1VP" --target "judges"
@REM echo training TMQ_NR_1VP_yf03_kfolds done.
@REM python train.py --name TMQ_NR_4VP_yf03_kfolds --src_root "D:\These\Projets\CompareMetrics\out\TMQ\New_Render\Y_fixed_0.3" --root_refPatches "\Source\4VP" --root_distPatches "\Distorted\4VP" --target "judges"
@REM echo training TMQ_NR_4VP_yf03_kfolds done.

pause