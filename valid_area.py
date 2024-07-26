#### this script is to erase alla the buildings that might be detected as change because of black image parts


import requests
import json
import orbital_vault as ov
from shapely import geometry, ops
from tqdm import tqdm
from pimsys.regions.RegionsDb import RegionsDb
import os
import pandas as pd
import rasterio
import numpy as np
import rasterio.features
import imageio


def get_polygons_from_changes(changes, poly, connectivity=4):

    height, width = changes.shape[0], changes.shape[1]
    factor=1
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)


    result = []
    for shape, value in rasterio.features.shapes(changes, connectivity=connectivity, transform=geotiff_transform):
        if value == 1:
            result.append(shape)

    return result


def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
    if channels>1:
     for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
    else:
       new_dataset.write(image, 1)
    new_dataset.close()

    return True



def get_wms_image_by_id(image_id, creds_mapserver, settings_db):
    image_broker_url = 'https://maps.orbitaleye.nl/image-broker/products?id={}&_skip=0'.format(image_id)
    response = requests.get(image_broker_url, auth=(creds_mapserver['username'], creds_mapserver['password']))
    wms_image = json.loads(response.text)[0]

    with RegionsDb(settings_db) as database:
        wms_image = database.get_optical_image_by_wms_layer_name(wms_image['wms_layer_name'])
        
    return wms_image

settings_db = ov.get_sarccdb_credentials()
creds_mapserver = ov.get_image_broker_credentials()


ids = os.listdir('./out_Fuel/B/')

#ids = ['data_region_9773089.p']

for _, id in enumerate(tqdm(ids)):
    print(id)
    pfile1 = pd.read_pickle('./out_Fuel/A/{}'.format(id))
    pfile2 = pd.read_pickle('./out_Fuel/B/{}'.format(id))

    img1_id = pfile1['image_id']
    img2_id = pfile2['image_id']

    wms_image_list_a = get_wms_image_by_id(img1_id, creds_mapserver, settings_db)
    wms_image_list_b = get_wms_image_by_id(img2_id, creds_mapserver, settings_db)
 
    valid_area_tot_a = ops.unary_union(wms_image_list_a['valid_area']).buffer(0.0)
    valid_area_tot_b = ops.unary_union(wms_image_list_b['valid_area']).buffer(0.0)

    valid_area_tot = valid_area_tot_a.intersection(valid_area_tot_b)

    pred = rasterio.open('./PIPELINE_RESULTS_Fuel/regularized/{}.tif'.format(pfile1['region_id']))

    pred = pred.read()
    pred = np.transpose(pred, (1,2,0))
    idx = np.where(pred!=[0,0,0])

    im_zeros = np.zeros((pred.shape[0], pred.shape[1]))
    im_zeros[idx[:2]]=255
    im_zeros = np.array(im_zeros, dtype=np.uint8)

    height, width = pred.shape[0], pred.shape[1]

    geotiff_transform = rasterio.transform.from_bounds(pfile1['poly'].bounds[0], pfile1['poly'].bounds[1],
                                                       pfile1['poly'].bounds[2], pfile1['poly'].bounds[3],
                                                       width, height)


    result = []

    for shape, value in rasterio.features.shapes(im_zeros, connectivity=4, transform=geotiff_transform):
        if value==255:
            result.append(shape)

    if result:
        del_polys = []
        for opol in result:
            opol = geometry.Polygon(opol['coordinates'][0])
            if opol.centroid.within(valid_area_tot):
               del_polys.append(opol)
            else:
                print('duhhhh')

        try:
          burned = rasterio.features.rasterize(shapes=del_polys, out_shape=(pred.shape[0], pred.shape[1]), all_touched=True, transform=geotiff_transform)
          idx0 = np.where(burned!=1)
          pred[:,:,0][idx0]=0
          pred[:,:,1][idx0]=0      
          pred[:,:,2][idx0]=0      
          save_tif_coregistered('./PIPELINE_RESULTS_Fuel/OUTPUT_valid/{}.tif'.format(pfile1['region_id']), pred, pfile1['poly'], channels=3)
        except:
          print('EXCEPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')
          save_tif_coregistered('./PIPELINE_RESULTS_Fuel/OUTPUT_valid/{}.tif'.format(pfile1['region_id']), pred, pfile1['poly'], channels=3)


    else:
        save_tif_coregistered('./PIPELINE_RESULTS_Fuel/OUTPUT_valid/{}.tif'.format(pfile1['region_id']), pred, pfile1['poly'], channels=3)

        print('else')


