import numpy as np
import cv2
import requests
from geopy.distance import geodesic
from shapely import geometry, wkb
from datetime import datetime
import torch
import clip
from PIL import Image as imp
from typing import Dict, Any
import rasterio
import base64
import json
import requests


def download_from_mapserver(image, region_bounds, auth=None):
    pixel_resolution_x = 0.5#layer["pixel_resolution_x"]
    pixel_resolution_y = 0.5#layer["pixel_resolution_y"]

    region_width = geodesic(
    (region_bounds[1], region_bounds[0]), (region_bounds[1], region_bounds[2])
    ).meters
    region_height = geodesic(
    (region_bounds[1], region_bounds[0]), (region_bounds[3], region_bounds[0])
    ).meters

    width = int(round(region_width / pixel_resolution_x))
    height = int(round(region_height / pixel_resolution_y))

    url = "https://maps.orbitaleye.nl/mapserver/?map=/maps/_"
    image_url = url + f'{image["wms_layer_name"]}.map&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX={region_bounds[0]},{region_bounds[1]},{region_bounds[2]},{region_bounds[3]}&WIDTH={width}&HEIGHT={height}&FORMAT=image/png&LAYERS={image["wms_layer_name"]}'
    resp = requests.get(image_url, auth=auth)
    if resp.ok:
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]]
    else: 
        print(image_url)
        return None
    
    return image



def search_tables(cur, query, image_id):
                cur.execute(query, (image_id,))
                data_tmp = cur.fetchall()
                desc = cur.description
                col_names = [x[0] for x in desc]
                results = [dict(zip(col_names, x)) for x in data_tmp]  

                return results



def clip_similarity(
    model: Any,
    preprocess: Any,
    image: np.ndarray,
    image_before: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Computes the similarity between two images using CLIP model

    Args:
        model (any): model used by CLIP
        preprocess (any): preprocessing function used by CLIP
        image (Image): image to be compared
        image_before (Image): reference image to be compared with for similarity

    Returns:
        float: similarity between the two images (0-1)
    """
    #print('aaaaaaaaaaaaaaaaaaaaa', image.shape, image_before.shape)

    image_p = preprocess(imp.fromarray(image)).unsqueeze(0).to(device)
    image_before_p = preprocess(imp.fromarray(image_before)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features1 = model.encode_image(image_p)
        image_features2 = model.encode_image(image_before_p)

    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    similarity = image_features2.cpu().numpy() @ image_features1.cpu().numpy().T

    return similarity


def save_tif_coregistered(filename, image, img_id, capture_timestamp, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    with rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform) as dst:
   # Write bands
        if channels>1:
         for ch in range(0, image.shape[2]):
           dst.write(image[:,:,ch], ch+1)
        else:
           dst.write(image, 1)

        dst.update_tags(image_id='{}'.format(img_id))
        dst.update_tags(capture_timestamp='{}'.format(capture_timestamp))

    dst.close()

    return True


def pass_image2docker(image_reference, image_target, filter_buildings_mask, test_url):
    image_ref_d = base64.b64encode(np.ascontiguousarray(image_reference))
    image_target_d = base64.b64encode(np.ascontiguousarray(image_target))
    filter_buildings_mask_d = base64.b64encode(np.ascontiguousarray(filter_buildings_mask))

    print('sending request to cd docker')
    response = requests.post(test_url, json={'image_reference':image_ref_d.decode(),'image_target':image_target_d.decode(), 'filter_buildings_mask':filter_buildings_mask_d.decode(), 'shape': json.dumps(list(image_reference.shape))})


    if response.ok:
        response_result = json.loads(response.text)
        if response_result['flag']==True:
            image = base64.b64decode(response_result['image'])
            result = np.frombuffer(image, dtype=np.uint8)
            result = np.reshape(result, image_reference.shape)
            return response_result['flag'], result
        else:
            return response_result['flag'], response_result['image']
