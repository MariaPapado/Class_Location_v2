from pimsys.regions.RegionsDb import RegionsDb
import orbital_vault as ov
from datetime import datetime
import shapely.geometry as geometry
from utils import *
import cv2
import psycopg
from tqdm import tqdm
import os
from PIL import Image
import rasterio.windows
import rasterio.features
import base64
import json
import requests


creds_mapserver = ov.get_image_broker_credentials()

creds = ov.get_sarccdb_credentials()

project_name = "National-Fuel-2024"


###JULY
#blank regions:  [5997030, 10308353, 10308525]
#interval_start = datetime(2024,6,16)
#interval_end = datetime(2024,7,15)

#MAY
interval_start = datetime(2024,5,16)
interval_end = datetime(2024,6,15)

connection_string = f"dbname={creds['database']} host={creds['host']} password={creds['password']} port={creds['port']} user={creds['user']}"

query0 = "SELECT id FROM public.optical_image WHERE wms_layer_name=%s"
query1 = "SELECT status FROM public.optical_image_coregistration WHERE image_id=%s"
query2 = """SELECT sar.region_id
FROM public.skywatch_aoi_request sar
join public.optical_image oi on oi.provider_reference = sar.aoi_id
WHERE oi.id = %s"""

dead_imgs = os.listdir('/cephfs/pimsys/coregistration/bad_tifs/production')
no_reg_tifs = [os.path.splitext(file)[0] for file in dead_imgs]

basemaps_tifs = os.listdir('/cephfs/pimsys/coregistration/basemaps/regions/')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("ViT-B/32", device=device)

blank_regions = []
bad_regs = []
total_chosen = 0
print('ok')
bad_regions = []
with psycopg.connect(connection_string) as conn:
    with conn.cursor() as cur:
        
        with RegionsDb(creds) as database:

            regions = database.get_regions_by_customer(project_name)
            multi_poly = geometry.MultiPolygon([x["bounds"] for x in regions]).buffer(0.0)  #just the bounds of the regions, no miages yet !!!!!!!!!!!!!! 

            regions_count = []
            for _, region in enumerate(tqdm(regions)):
                print(region['id'])
                images = database.get_optical_images_containing_point_in_period([region['bounds'].centroid.x, region['bounds'].centroid.y], [int(interval_start.timestamp()), int(interval_end.timestamp())])  ##can be improved!!
                wms_images = sorted(images, key=lambda x: x["capture_timestamp"])
                wms_images = [x for x in wms_images if x["source"] != "Sentinel-2"]

#                bounds_window = rasterio.features.bounds(region['bounds'].bounds)
                ref_img_file = rasterio.open('/cephfs/pimsys/coregistration/basemaps/regions/region_{}.tif'.format(region['id']))
                window = rasterio.windows.from_bounds(*region['bounds'].bounds, ref_img_file.transform)
                ref_img = ref_img_file.read(window=window, boundless=True)
                ref_img = np.transpose(ref_img, (1,2,0))

                reg_good = []
                for i, wms_im in enumerate((wms_images)):

                    image_id = search_tables(cur, query0, wms_im['wms_layer_name'])[0]['id']
                    wms_im['image_id'] = image_id
                    reg_status = search_tables(cur, query1, image_id)
                    if len(reg_status)!=0 and reg_status[0]['status'] == 'ml_good':
                        reg_status = [{'status': 'ml_good'}]
                        reg_good.append(wms_im)
                    
                    #print(wms_im['wms_layer_name'])

                    if len(reg_status)==0:
                        if wms_im['wms_layer_name'][0] in no_reg_tifs:
                            reg_status = [{'status': 'ml_bad'}]
                        else:
                            reg_status = [{'status': 'ml_good'}]
                            reg_good.append(wms_im)

                    wms_im['reg_status'] = reg_status[0]['status']
                    #break
                

                regions_count.append(len(wms_images))

                blank_flag = False
                best_img = []
                if len(reg_good)==1:
                    #target_img = download_from_mapserver(reg_good[0], region['bounds'].bounds, (creds_mapserver['username'], creds_mapserver['password']))
                    best_img = reg_good[0]


                elif len(reg_good)>1:
                    measure_max = 0
                    best_img = reg_good[0]
                    for next_img in reg_good:
                        target_img = download_from_mapserver(next_img, region['bounds'].bounds, (creds_mapserver['username'], creds_mapserver['password']))
                        measure = clip_similarity(
                            model,
                            preprocess,
                            # This is where the actual image is passed to the model
                            ref_img.astype(np.uint8),
                                target_img.astype(np.uint8),
                                device).item()
                        if measure >measure_max:
                            measure_max = measure
                            best_img = next_img
                    #print('mmax', measure_max)

                    
                else:
                    bad_regions.append(region['id'])
                    blank_flag = True
 
                if blank_flag==False:
                    check_good = search_tables(cur, query2, best_img['image_id'])
                    if check_good[0]['region_id'] > 0:
                        target_img = download_from_mapserver(best_img, region['bounds'].bounds, (creds_mapserver['username'], creds_mapserver['password']))
                        img_id_save = best_img['image_id']
                        total_chosen = total_chosen + 1
                    elif check_good[0]['region_id'] < 0:
                        target_img = download_from_mapserver(best_img, region['bounds'].bounds, (creds_mapserver['username'], creds_mapserver['password']))
                        img_id_save = best_img['image_id']

                        filter_buildings_mask_file = rasterio.open('/cephfs/pimsys/coregistration/basemaps/regions_buildings/region_{}.tif'.format(region['id']))
                        window = rasterio.windows.from_bounds(*region['bounds'].bounds, filter_buildings_mask_file.transform)
                        filter_buildings_mask = filter_buildings_mask_file.read(window=window, boundless=True)

                        filter_buildings_mask = np.resize(filter_buildings_mask[0], (target_img.shape[0], target_img.shape[1]))
                        

                        ref_img = cv2.resize(ref_img, (target_img.shape[1], target_img.shape[0]), cv2.INTER_NEAREST)
                        #print('shape', ref_img.shape, target_img.shape, filter_buildings_mask.shape)

                        filter_buildings_mask = filter_buildings_mask > 0   ######CHECK AGAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        
                        flag_doc, repeat_reg_img = pass_image2docker(ref_img, target_img, filter_buildings_mask, 'http://10.10.100.8:8062/api/process')

                        ########################xmmmm na kanw h auto h na vrw prwta tis eikones ksexwrista gia tous 2 mhnes kai meta !!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                        ###########apla na tyis sugkrinw me ena registration docker ksexwrista... xmmmm!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                        if flag_doc==True:
                            print('YESSSSSSSSS')
                            target_img = repeat_reg_img
                        else:
                            bad_regions.append(region['id'])                        

                    save_tif_coregistered('{}/{}.tif'.format('./May_images/',region['id']), target_img, img_id_save, geometry.Polygon.from_bounds(region['bounds'].bounds[0], region['bounds'].bounds[1], region['bounds'].bounds[2], region['bounds'].bounds[3]), channels = 3)
                else:
                    print('BLANK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            print('total regions: ', len(regions_count))
            print('bad regions: ', len(bad_regions))
            print(bad_regions)   


            #bad regions July[9750375, 5996484, 5996485, 5996652, 5996653, 5996654, 5996655, 5996656, 9750665, 5996877, 5996884, 5997030, 5997035, 5997040, 5997041, 5997045, 5997047, 5997191, 5997193, 5997200, 
            #5997570, 5997741, 5997914, 5998093, 5998094, 5998100, 5998281, 5998452, 9752183, 5999177, 5999345, 5999515, 10308067, 10308068, 10308353, 10308525, 10309173, 10309654, 10310668]

#bad regions May [5996496, 5996497, 5996498, 9750384, 5996642, 5996643, 5996644, 5996645, 5996646, 5996665, 5996666, 5996667, 5996668, 5996669, 5996670, 5996671, 5996672, 9750527, 5996811, 5996812, 5996813, 5996814, 5996815, 5996838,
# 5996839, 5996840, 5996841, 5996842, 5996843, 5996844, 5996845, 5996846, 5996847, 9750682, 5996877, 5996980, 9750805, 5997013, 5997014, 5997015, 5997016, 5997017, 5997018, 5997019, 5997020, 5997021, 5997022, 5997023, 
# 5997024, 5997030, 5997031, 5997032, 5997033, 5997034, 5997035, 5997036, 5997037, 5997038, 5997039, 5997040, 5997041, 5997042, 5997043, 5997044, 5997045, 5997046, 5997047, 5997048, 5997149, 9750968, 5997188, 5997189, 
# 5997191, 5997192, 5997193, 5997194, 5997195, 5997196, 5997197, 5997198, 5997199, 5997200, 5997201, 9751083, 5997487, 9751248, 5997657, 5997825, 5997826, 5997920, 5997994, 5997995, 5998097, 5998099, 5998100, 5998103, 
# 5998104, 5998105, 9751820, 9751981, 5998270, 5998272, 5998273, 5998274, 5998275, 5998276, 5998277, 5998278, 5998279, 5998280, 5998281, 5998282, 5998283, 5998449, 5998450, 5998451, 5998452, 5998453, 5998454, 5998461,
#  5998462, 9752311, 5998630, 5998635, 5998636, 5998638, 5998639, 5999844, 10308074, 10308216, 10308369, 10308387, 10308388, 10308389, 10308390, 10308391, 10308392, 10308393, 10308395, 10308396, 10308397, 10308398, 
#  10308520, 10308521, 10308522, 10308523, 10308524, 10308525, 10308651, 10308652, 10308653, 10308661, 10308662, 10308663, 10309497, 10309498, 10309655, 10309657, 10309658, 10309818, 10309819, 10309820, 
#  10309821, 10309822, 10309823, 10309824, 10309825, 10309982, 10309983, 10310979]

#NEG July images to check visually @@@@@@@@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!! for friday 
#9750375
#5996485
#5996654
#5996656

#    print (how many images are registered correctly and how many images ar note)  ##HOMEWORK!

