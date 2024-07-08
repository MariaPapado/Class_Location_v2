import sys
sys.path.append('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/')

import torch
from PIL import Image
import numpy as np
import cv2
import os
import rasterio
import shutil
from tools import *
from shapely import geometry
#from buildings import *
from tqdm import tqdm
from mamba_class import Trainer
import argparse


parser = argparse.ArgumentParser(description="Training on xBD dataset")
parser.add_argument(
        '--cfg', type=str, default='./changedetection/configs/vssm1/vssm_small_224.yaml')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--pretrained_weight_path', type=str)

parser.add_argument('--dataset', type=str, default='xBD')
parser.add_argument('--type', type=str, default='train')
parser.add_argument('--dataset_path', type=str, default = '/home/mariapap/CODE/ChangeOS/REGIONS_August/')
#                        default='/home/mariapap/CODE/ChangeOS/DEQUAN/test/after/')
#    parser.add_argument('--train_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt')
#    parser.add_argument('--test_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/test')
#    parser.add_argument('--test_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/val_all.txt')
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--crop_size', type=int, default=512)
parser.add_argument('--train_data_name_list', type=list)
parser.add_argument('--test_data_name_list', type=list)
parser.add_argument('--start_iter', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--max_iters', type=int, default=240000)
parser.add_argument('--model_type', type=str, default='STMambaBCD')
parser.add_argument('--model_param_path', type=str, default='')

args = parser.parse_args()



#/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/images/A
#################################################################################################################################

mamba_format = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml'
#mamba_weights = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/changedetection/saved_models/dequan_71250_model.pth'
mamba_weights = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/changedetection/saved_models/frp_8298_8255_8342_model.pth'






#################################################################################################################################
dir_before = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/images/A/'
#dir_after = '/home/mariapap/CODE/ChangeOS/REGIONS_Dec/'
dir_after = '/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/images/B/'

ids = os.listdir(dir_before)

####folders
f_pipeline_result = './PIPELINE_RESULTS_NAT'
f_before_registered = f_pipeline_result + '/BEFORE_REGISTERED'
f_pred_before = f_pipeline_result + '/PREDS_BEFORE'
f_pred_after = f_pipeline_result + '/PREDS_AFTER'
f_output = f_pipeline_result + '/OUTPUT'
#f_output_disappear = f_pipeline_result + '/OUTPUT_DISAPPEAR'
##############


if os.path.exists(f_pipeline_result):
    shutil.rmtree(f_pipeline_result)
os.mkdir(f_pipeline_result)

#os.mkdir(f_before)
#os.mkdir(f_after)
os.mkdir(f_before_registered)
os.mkdir(f_pred_before)
os.mkdir(f_pred_after)
os.mkdir(f_output)

#ids = ['9352495_399.tif', '9296660_1.tif', '9316019_127.tif', '9356016_423.tif', '9349479_378.tif', '9403558_764.tif', '9384195_631.tif', '9379913_595.tif']

for _, id in enumerate(tqdm(ids)):
    print(id)

    raster_before = rasterio.open(dir_before + id)
    raster_after = rasterio.open(dir_after +  id)
#    raster_after = np.array(raster_after)
#    print('aaaa', raster_before.shape, raster_after.shape)

    im_before, im_after, bounds = check_shape_and_resize(raster_before, raster_after)


#    print('AAAAAAAAAAAAAAAAAAAAAAAAAA', im_before.shape, im_after.shape)



    try:
        im_before_transformed, flag = register_image_pair(im_after, im_before)
    except:
        im_before_transformed, flag = im_after

#    im_before = histogram_match(im_before, im_after_transformed)
#    im_after_transformed = histogram_match(im_after_transformed, im_before)

    if flag==True:

        save_tif_coregistered('{}/before_transformed_{}'.format(f_before_registered,id), im_before_transformed, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)
    #    save_tif_coregistered('{}/before_{}'.format(f_after_registered,id), im_before, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)


        trainer = Trainer(mamba_format, mamba_weights, args)


        buildings_before = trainer.validation(im_before_transformed)
        buildings_before = filter_small_contours(buildings_before)
    

        buildings_after = trainer.validation(im_after)
        buildings_after = filter_small_contours(buildings_after)
    ##############################################################################################################################################################
    #################################################################################################################################################################

    #    model = torch.jit.load('/home/mariapap/CODE/ChangeOS/models/changeos_r101.pt')
    #    model.eval()
    #    model = ChangeOS(model)

    #    patch_size, step = 512, 512
    #    buildings_before, buildings_after = make_prediction(im_before, im_after_transformed, model, patch_size, step, id)
        save_tif_coregistered('{}/{}'.format(f_pred_before,id), buildings_before*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)
        save_tif_coregistered('{}/{}'.format(f_pred_after,id), buildings_after*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)


        output_appears = apply_watershed(buildings_before, buildings_after)
        output_disappears = apply_watershed(buildings_after, buildings_before)

        idx2 = np.where(output_disappears==1)
        output_appears[idx2]=2

        output = visualize(output_appears)

        save_tif_coregistered('{}/output_{}'.format(f_output,id), output, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)












    
