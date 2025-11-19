# Ce script python est créé pour alléger le projet. L'objectif est de ne pas avoir besoin de stocker toute la BDD patchifiée,
# mais de créer les patches en mémoire sans les enregistrer, et garder les valeurs des LPIPS dans un tableau correspondant à la position de chaque patch.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ENG : This python script is created to lighten the project. The objective is to not have to store the entire patchified database,
# The difference with original is that the patches are not saved in the disk, but are created in memory. Saving a lot of space and time.

import argparse
import os
import lpips
import torch
import numpy as np
import statsmodels.api as sm
import cv2
from scipy import stats
import csv
from itertools import groupby
from operator import itemgetter
from statistics import mean
from decimal import Decimal
import find_dis_ref
import correlation_VP
import re

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--use_gpu', action='store_true', default=True, help='turn on flag to use GPU')
parser.add_argument('--version', type=str, default='0.1')
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-use_folds', type=bool, required=True)
parser.add_argument('-v', '--views', type=int, required=True)
parser.add_argument('-vm', '--view_method', type=str, required=True)
parser.add_argument('-rm', '--render_method', type=str, required=True)
parser.add_argument('-db', '--database', type=str, required=True)
parser.add_argument('-mos', '--mos_csv_file', type=str, required=True)
parser.add_argument('-testlist', '--test_list_csv', type=str, required=True)
opt = parser.parse_args()

model = opt.model
modelpath = './checkpoints/'+ model +'/latest_net_.pth'
use_folds = opt.use_folds
testing_views = opt.views
view_method = opt.view_method
render_method = opt.render_method
database = opt.database
mos_csv_file = opt.mos_csv_file
test_list_csv = opt.test_list_csv
opt = parser.parse_args() 
force_overwrite = False # If the file already exists, we will overwrite it. If False, we will not overwrite it.
out = './out/' + database + '/' + render_method + '/' + view_method + '/' + model + '/' + str(testing_views) + 'VP/'
# ------------------------------- DEBUG VARIABLES -------------------------------
root_refPatches = 'D:/These/Projets/CompareMetrics/out/' + database + '/' + render_method + '/' + view_method + '/Source/' + str(testing_views) + 'VP/' 
root_disPatches = 'D:/These/Projets/CompareMetrics/out/' + database + '/' + render_method + '/' + view_method + '/Distorted/' + str(testing_views) + 'VP/'
ext = '.png'
# The 20% is the one that is used in the test_list_csv file.
# ref_obj_list = find_dis_ref.find_ref_files(root_refPatches)
if (use_folds):
    # there are  folds in the test list csv file. Names are like TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6_k0.csv, TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6_k1.csv, ...
    ref_obj_list_folds = []
    model_folds = []
    output_folds = []
    for fold in range(5):
        # ref_obj_list = correlation_VP.get_testset_ref_list(test_list_csv) # We will take the reference objects from the CSV file. The function 'get_ref_obj_list' will return a list of the reference objects.
        # For folds, we transform 
        # './dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv' into './dataset/folds/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6_k${fold}.csv'
        if database == "TMQ":
            test_list_csv_fold = './dataset/folds/' + test_list_csv.split('\\')[-1].replace('.csv', '_k' + str(fold) + '.csv')
        else: 
            test_list_csv_fold = './dataset/TSMD/_TSMD_fulldataset.csv'#'./dataset/TSMD/folds/' + test_list_csv.split('\\')[-1].replace('.csv', '_k' + str(fold) + '.csv')
            
        print("Using test list CSV file for fold %d: %s" % (fold, test_list_csv_fold))
        ref_obj_list_folds.append(correlation_VP.get_testset_ref_list(test_list_csv_fold))
        # the model is also different
        # e.g. : './checkpoints/TMQ_NR_1VP_org_TAA/fold_k0/latest_net_.pth'
        model_folds.append('./checkpoints/' + model + '/fold_k' + str(fold) + '/latest_net_.pth')
        # Modifying the output folder to add the fold number
        output_folds.append(out + 'fold_k' + str(fold) + '/')
        
else:
    ref_obj_list_folds = [correlation_VP.get_testset_ref_list(test_list_csv)]
    model_folds = [modelpath]
    print("Avaluating the testset with %s model" % (model))
    output_folds = [out]


# patches_csv_list = find_dis_ref.find_ref_csvfiles(root_refPatches)
#views_folder = '..../out/ref_folder/obj_name/views'



# loss_fn = lpips.LPIPS(net='alex', version=opt.version, model_path=modelpath)
# if opt.use_gpu:
#     loss_fn = loss_fn.cuda()
# loss_fn.eval()
# torch.set_grad_enabled(False)


List_MOS = []
for fold_idx, ref_obj_list in enumerate(ref_obj_list_folds):
    loss_fn = lpips.LPIPS(net='alex',version=opt.version, model_path = model_folds[fold_idx])# e.g. model_path = './checkpoints/Trial1/latest_net_.pth'
    if(opt.use_gpu):
        loss_fn.cuda()
    sd = loss_fn.state_dict()
    # print("Fold", fold_idx, "sum weights:", sum(v.abs().sum().item() for v in sd.values()))
    sd = loss_fn.state_dict()
    print("CKPT loaded keys:", len(sd))
    for k in ["lins.0.model.1.weight","net.slice1.0.weight"]:
        if k in sd: print(k, float(sd[k].abs().sum()))
    ## Output file
    #If the file already exists, we delete it
    # else we create it
    if not(os.path.exists(output_folds[fold_idx])):
        os.makedirs(os.path.dirname(output_folds[fold_idx]), exist_ok=True)    
    print('--- Starting fold k%d ---'%fold_idx)
    for ref_obj in ref_obj_list:

        ref_views_folder = root_refPatches + '/' + ref_obj + '/views'
        
        distorted_obj_list = find_dis_ref.find_dis_files(root_disPatches, ref_obj)
        # Creating output folder for reference object
        # For each reference object, we create a folder that contains all the output csv files of the distorted objects
        currentFolder = output_folds[fold_idx] + ref_obj + '/'
        
        # if not(os.path.exists(currentFolder)):
        #     os.makedirs(os.path.dirname(currentFolder), exist_ok=True)

        if not(os.path.exists(output_folds[fold_idx] + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/')):
            os.makedirs(os.path.dirname(output_folds[fold_idx] + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/'), exist_ok=True)    
            print('Creating the folder %s' % (output_folds[fold_idx] + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/'))

        if(os.path.exists(output_folds[fold_idx] + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results_testset.csv') and force_overwrite == False):
            print('The file %s already exists. We will not overwrite it.' % (output_folds[fold_idx] + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results.csv'))
            continue
        
        print('Creating the file %s' % (output_folds[fold_idx] + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results_testset.csv'))
        file_GLPIPS = open(output_folds[fold_idx] + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results_testset.csv','w')
        file_GLPIPS.writelines('ObjectName, MOS, LPIPS\n')

        
        for distorted_obj in distorted_obj_list:
            # Finding the csv file of the ref object
            List_GraphicsLPIPS = []
            # Finding the csv file of the distorted object        
            outcsvfile = currentFolder + distorted_obj + '_LGLPIPS_scores.csv'
            # if(os.path.exists(outcsvfile) and force_overwrite == False):
            #     print('The file %s already exists. We will not overwrite it.' % outcsvfile)
            #     continue
            # f = open(outcsvfile,'w')
            
            dis_views_folder = root_disPatches + '/' + distorted_obj + '/views'
            # print("find dis ref : ", root_refPatches  + ref_obj)
            csv_patch_file = find_dis_ref.find_ref_csvfiles(root_refPatches + ref_obj)[0]

            ###--------------------DEBUG START--------------------###
            # if(correlation_VP.get_MOS(mos_csv_file, distorted_obj, 2, 3) == -1): 
            # # if(correlation_VP.get_test_MOS(test_list_csv, distorted_obj) == -1): # ONLY FOR TMQ 
            #     print('[DEBUG] The object %s is not in the MOS file. We will skip it.' % distorted_obj)
            #     continue
            ###---------------------DEBUG END---------------------###
            # Creating the output csv file for the distorted object
            List_MOS.append([correlation_VP.get_MOS(mos_csv_file, distorted_obj, name_col = 0, mos_col = 1)]) # WARNING : MOS_COL and NAME_COL needs to be specified correctly
            # List_MOS.append([correlation_VP.get_test_MOS(test_list_csv, distorted_obj)])

            with open(csv_patch_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                v = 1  # Current view point
                res = []

                for row in csv_reader:
                    if line_count == 0:
                        stepX = int(row[2].split('=')[1].strip())
                        stepY = int(row[3].split('=')[1].strip())
                        patchSize = int(row[4].split('=')[1].strip())
                        overlapThreshold = float(row[5].split('=')[1].strip())
                        objectName = row[6].split('=')[1].strip()
                        nbPatchesVn = [int(r.split('=')[1].strip()) for r in row[7:]]
                        vn = len(nbPatchesVn)

                        # Chargement des deux premières images (view_1)
                        refimg = cv2.imread(f"{ref_views_folder}/view_{v}{ext}")[:, :, ::-1]
                        disimg = cv2.imread(f"{dis_views_folder}/view_{v}{ext}")[:, :, ::-1]

                        patches0 = []
                        patches1 = []
                        patch_counter = 0
                    else:
                        if line_count > sum(nbPatchesVn[0:v]):
                            # Nouveau viewpoint, calcul du batch précédent
                            if patches0:
                                batch0 = torch.cat([lpips.im2tensor(p).cuda() for p in patches0], dim=0)
                                batch1 = torch.cat([lpips.im2tensor(p).cuda() for p in patches1], dim=0)
                                with torch.no_grad():
                                    dists_t = loss_fn(batch0, batch1)           # [N] ou [N,1]
                                    dists_t = dists_t.view(-1)
                                    dists_np = dists_t.detach().cpu().numpy()
                                    np.clip(dists_np, 0.0, 1.0, out=dists_np)
                                GraphicsLPIPS = float(dists_np.mean())
                                List_GraphicsLPIPS.append(GraphicsLPIPS)

                            # Nouveau point de vue
                            v += 1
                            refimg = cv2.imread(f"{ref_views_folder}/view_{v}{ext}")[:, :, ::-1]
                            disimg = cv2.imread(f"{dis_views_folder}/view_{v}{ext}")[:, :, ::-1]
                            patches0 = []
                            patches1 = []

                        x, y = int(row[0]), int(row[1])
                        patch0 = refimg[y:y+patchSize, x:x+patchSize]
                        patch1 = disimg[y:y+patchSize, x:x+patchSize]
                        if patch0.shape[:2] != (patchSize, patchSize) or patch1.shape[:2] != (patchSize, patchSize):
                            continue  # évite les erreurs bord image
                        patches0.append(patch0)
                        patches1.append(patch1)

                    line_count += 1

                # Dernier viewpoint : calcul final
                if patches0:
                    batch0 = torch.cat([lpips.im2tensor(p).cuda() for p in patches0], dim=0)
                    batch1 = torch.cat([lpips.im2tensor(p).cuda() for p in patches1], dim=0)
                    with torch.no_grad():
                        dists = loss_fn(batch0, batch1).view(-1).cpu().numpy()
                        np.clip(dists, 0.0, 1.0, out=dists)
                    GraphicsLPIPS = dists.mean()
                    List_GraphicsLPIPS.append(GraphicsLPIPS)

            # f.close()
            List_MOS[-1].append(List_GraphicsLPIPS)
            # List_MOS looks like this : [[MOS, [LPIPS]], [MOS, [LPIPS]], ...]
            # print('writing the file %s' % file_GLPIPS.name)
            file_GLPIPS.writelines('%s, %.2f, ' % (distorted_obj, List_MOS[-1][0]))
            for i in range(len(List_GraphicsLPIPS)):
                file_GLPIPS.writelines('%.6f' % List_GraphicsLPIPS[i])
                if(i != len(List_GraphicsLPIPS)-1):
                    file_GLPIPS.writelines(', ')
            file_GLPIPS.writelines('\n')
        file_GLPIPS.close()
        
    #     Graphicslpips = sum(res)/len(res)
    #     List_GraphicsLPIPS.append(Graphicslpips)
    #     # List_MOS.append(float(MOS))
    #     List_GraphicsLPIPS = np.array(List_GraphicsLPIPS)
    #     # List_MOS = np.array(List_MOS)

    #     List_GraphicsLPIPS = sm.add_constant(List_GraphicsLPIPS)
