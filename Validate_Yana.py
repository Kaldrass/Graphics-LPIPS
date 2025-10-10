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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--csvfile', type=str, default='./dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv')
parser.add_argument('-m','--modelpath', type=str, default='./checkpoints/TMQ_OR_1VPn_org/latest_net_.pth', help='location of model')
parser.add_argument('-o','--out', type=str, default='./out/TMQ_Yana_ORG/', help='output folder. Do not forget to add the name of the database and end with \'/\'.')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', default=True, help='turn on flag to use GPU')

opt = parser.parse_args()
force_overwrite = False # If the file already exists, we will overwrite it. If False, we will not overwrite it.
# ------------------------------- DEBUG VARIABLES -------------------------------
   
ext = '.png'
mos_csv_file = 'D:/These/BDD/TMQ/Collected_Data/MOS+CI_3000stimuli.csv' # It takes the whole true MOS dataset. In fact the metrics is trained on 80% of it and tested on 20%.
# The 20% is the one that is used in the test_list_csv file.
test_list_csv = r'D:\These\Graphics-LPIPS\dataset\TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv' # We need to take the 1st column of the CSV file as the list of files.
# ref_obj_list = find_dis_ref.find_ref_files(root_refPatches)
ref_obj_list = correlation_VP.get_testset_ref_list(test_list_csv) # We will take the reference objects from the CSV file. The function 'get_ref_obj_list' will return a list of the reference objects.
# dis_obj_list = correlation_VP.get_testset_dis_list(test_list_csv) # We will take the distorted objects from the CSV file. The function 'get_dis_obj_list' will return a list of the distorted objects.
# ------------------------------- YANA VALIDATION VARIABLES -------------------------------
root_vp1_dis = "D:/These/BDD/TMQ/Subset of 3000 stimuli/Snapshots_Distorted_Stimuli/VP1"
root_vp1_ref = "D:/These/Projets/CompareMetrics/out/TMQ/Old_Render/Original/Source/1VP"

# root_vp1_ref = "D:/These/BDD/TMQ/Subset of 3000 stimuli/Snapshots_Reference/VP1"



# patches_csv_list = find_dis_ref.find_ref_csvfiles(root_refPatches)
#views_folder = '..../out/ref_folder/obj_name/views'

loss_fn = lpips.LPIPS(net='alex',version=opt.version, model_path = opt.modelpath)# e.g. model_path = './checkpoints/Trial1/latest_net_.pth'
if(opt.use_gpu):
    loss_fn.cuda()
    print('Using GPU')

    
## Output file
#If the file already exists, we delete it
# else we create it
if not(os.path.exists(opt.out)):
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)    

List_MOS = []
for ref_obj in ref_obj_list: 
    

    obj_path = os.path.join(root_vp1_ref, ref_obj).replace('\\', '/')
    dis_obj_list = correlation_VP.get_testset_dis_list_from_ref(test_list_csv, ref_obj) 

    # distorted_obj_li  st = find_dis_ref.find_dis_files(root_disPatches, ref_obj)
    # Creating output folder for reference object
    # For each reference object, we create a folder that contains all the output csv files of the distorted objects
    currentFolder = opt.out + ref_obj + '/'
    
    # if not(os.path.exists(currentFolder)):
    #     os.makedirs(os.path.dirname(currentFolder), exist_ok=True)
    
    if not(os.path.exists(opt.out + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/')):
        os.makedirs(os.path.dirname(opt.out + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/'), exist_ok=True)    
        print('Creating the folder %s' % (opt.out + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/'))

    if(os.path.exists(opt.out + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results_testset.csv') and force_overwrite == False):
        print('The file %s already exists. We will not overwrite it.' % (opt.out + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results.csv'))
        continue
    
    print('Creating the file %s' % (opt.out + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results_testset.csv'))
    file_GLPIPS = open(opt.out + '_METRIC_RESULTS_TESTSET_/' + ref_obj + '/GLPIPS_results_testset.csv','w')
    file_GLPIPS.writelines('ObjectName, MOS, LPIPS\n')

    
    for distorted_obj in dis_obj_list:
        # Finding the csv file of the ref object
        List_GraphicsLPIPS = []
        # Finding the csv file of the distorted object        
        outcsvfile = currentFolder + distorted_obj + '_LGLPIPS_scores.csv'
        # if(os.path.exists(outcsvfile) and force_overwrite == False):
        #     print('The file %s already exists. We will not overwrite it.' % outcsvfile)
        #     continue
        # f = open(outcsvfile,'w')
        
        csv_patch_file = find_dis_ref.find_ref_csvfiles(root_vp1_ref + '/' + ref_obj)[0]

        ###--------------------DEBUG START--------------------###
        # if(correlation_VP.get_MOS(mos_csv_file, distorted_obj) == -1): #Those are the objects that are not in the MOS file : GOLDEN UNITS ?
        if(correlation_VP.get_test_MOS(test_list_csv, distorted_obj) == -1): #Those are the objects that are not in the MOS file : GOLDEN UNITS ? TEST_SET ?
            print('[DEBUG] The object %s is not in the MOS file. We will skip it.' % distorted_obj)
            continue
        ###---------------------DEBUG END---------------------###
        # Creating the output csv file for the distorted object
        # List_MOS.append([correlation_VP.get_MOS(mos_csv_file, distorted_obj)])
        List_MOS.append([correlation_VP.get_test_MOS(test_list_csv, distorted_obj)])
        
        with open(csv_patch_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            v = 1 # Current view point
            res = []
            resString =''

            for row in csv_reader:
                if line_count == 0:
                    # Gathering the information of the csv file
                    stepX = int(row[2].split('=')[1].strip())
                    stepY = int(row[3].split('=')[1].strip())
                    patchSize = int(row[4].split('=')[1].strip())
                    overlapThreshold = float(row[5].split('=')[1].strip())
                    objectName = row[6].split('=')[1].strip()
                    nbPatchesVn = []
                    for i in range(7, len(row)):
                        nbPatchesVn.append(int(row[i].split('=')[1].strip()))
                    vn = len(nbPatchesVn)

                    # f.writelines('x, y, score, stepX = %d, stepY = %d, patchSize = %d, overlapThreshold = %f' % (stepX, stepY, patchSize, overlapThreshold))
                    # # Adding ViewPoint information
                    # for i in range(1, vn+1):
                    #     f.writelines(', nbPatchesV%d = %d' % (i, nbPatchesVn[i-1]))
                    # f.writelines('\n')


                    refimg = os.path.join(obj_path, 'views', 'view_%d.png' % v).replace('\\', '/')
                    disimg = os.path.join(root_vp1_dis, distorted_obj + '.png').replace('\\', '/')
                    img0 = cv2.imread(refimg)[:,:,::-1]
                    img1 = cv2.imread(disimg)[:,:,::-1]
                    

                else:
                    # v is the view number. v += 1 when line_count > sum(nbPatchesVn[v], nbPatchesVn[v-1], ... up to 0)
                    if line_count > sum(nbPatchesVn[0:v]):
                        
                        GraphicsLPIPS = sum(res)/len(res)
                        List_GraphicsLPIPS.append(GraphicsLPIPS)
                        res = []
                        v += 1
                        refimg = obj_path
                        disimg = os.path.join(root_vp1_dis, distorted_obj + '.png')
                        # We load the images
                        img0 = cv2.imread(refimg)[:,:,::-1]
                        img1 = cv2.imread(disimg)[:,:,::-1]
                        
                    x = row[0]
                    y = row[1]
                    # f.writelines('%s, %s, ' % (x, y))
                    patch0 = img0[int(y):int(y)+patchSize, int(x):int(x)+patchSize]
                    patch1 = img1[int(y):int(y)+patchSize, int(x):int(x)+patchSize]
                    

                    # Now we exctract the patches from the images with the informations given in the csv file
                    # We will compare the patches of the distorted image with the patches of the reference image
                    # We will store the LPIPS values in a list
                    patch0 = lpips.im2tensor(patch0) # RGB image from [-1,1]
                    patch1 = lpips.im2tensor(patch1)
                        
                    if(opt.use_gpu):
                        patch0 = patch0.cuda()
                        patch1 = patch1.cuda()
                    dist01 = loss_fn.forward(patch0,patch1).reshape(1,).item()
                    if dist01 > 1:
                        dist01 = 1
                    # f.writelines('%.6f\n' %dist01)
                    
                    res.append(dist01)
                line_count += 1
            # For last view point, we need to compute the LPIPS score and add it to the list
            if len(res) > 0: # Should be always true but to avoid error
                GraphicsLPIPS = sum(res)/len(res)
                List_GraphicsLPIPS.append(GraphicsLPIPS)
                res = []
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
