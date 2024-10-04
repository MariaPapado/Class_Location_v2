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
from shapely.geometry import box, Polygon


import cosmic_eye_client
from CustomerDatabase import CustomerDatabase

from shapely import geometry, ops, wkb
import pyproj

def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((np.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return int(float(epsg_code))
    epsg_code = '327' + utm_band
    return int(float(epsg_code))

def get_corridor(pipelines, buffer=50):
    # get pipes
    # pipelines = get_pipelines(settings_client)
    # Get buffered geometry
    simplify = 5.0
    corridor = []
    for pipe in pipelines:
        lon, lat = pipe.centroid.xy
        utm_epsg = convert_wgs_to_utm(lon[0], lat[0])
        proj = pyproj.Transformer.from_crs(4326, utm_epsg, always_xy=True)
        proj_inverse = pyproj.Transformer.from_crs(utm_epsg, 4326, always_xy=True)
        pipe_tmp = ops.transform(proj.transform, pipe)
        pipe_buffer = pipe_tmp.buffer(buffer)
        corridor.append(pipe_buffer.simplify(simplify))
    # Stitch results
    corridor = geometry.MultiPolygon(corridor).buffer(0.0)
    corridor_4326 = ops.transform(proj_inverse.transform, corridor)

    return corridor_4326.buffer(0.0)


def get_polygons_from_changes(changes, poly, connectivity=4):

    height, width = changes.shape[0], changes.shape[1]
    factor=1
    geotiff_transform = rasterio.transform.from_bounds(bounds[0], bounds[1],
                                                       bounds[2], bounds[3],
                                                       width/factor, height/factor)


    result = []
    for shape, value in rasterio.features.shapes(changes, connectivity=connectivity, transform=geotiff_transform):
        if value == 1:
            result.append(shape)

    return result


def save_tif_coregistered_with_params(filename, image, xparams, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly[0], poly[1],
                                                       bounds[2], poly[3],
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

        dst.update_tags(**xparams)
#        dst.update_tags(img_id_after='{}'.format(xparams['id_before']))
        
    dst.close()

    return True


project_name = 'National-Fuel-2024'

customer_db_creds = ov.get_customerdb_credentials()
customer_db = CustomerDatabase(customer_db_creds['username'], customer_db_creds['password'])

project = customer_db.get_project_by_name(project_name)

ce_login = ov.get_project_server_credentials(project.get("name"), project.get("servers")[0].get("name"))
client = cosmic_eye_client.connect(project.get("servers")[0].get("domain"))
client.login(ce_login['user'], ce_login['password'])

pipe_contours = client.call_remote_method("getAllPipelines", [])

pipes = []
for pipeline in pipe_contours:
    points = wkb.loads(str(pipeline[3]), hex=True).coords
    pipe_line = geometry.LineString(np.array(points.xy).T)
    pipes.append(pipe_line)
    

corridor = get_corridor(pipes, buffer=project['pipeline_monitoring_distance'])
#print(corridor)


ids = os.listdir('./PIPELINE_RESULTS/OUTPUT_valid/')

xparams={}
for _, id in enumerate(tqdm(ids)):
    print(id)
    img = rasterio.open('./PIPELINE_RESULTS/OUTPUT_valid/{}'.format(id))
    bounds = img.bounds

    xparams['id_before'] = img.tags()['id_before']
    xparams['id_after'] = img.tags()['id_after']

    xparams['ctime_before'] = img.tags()['ctime_before']
    xparams['ctime_after'] = img.tags()['ctime_after']

    img = img.read()
    img = np.transpose(img, (1,2,0))
    #print(img.shape)

    width = img.shape[1]
    height = img.shape[0]
    geotiff_transform = rasterio.transform.from_bounds(bounds[0], bounds[1],
                                                       bounds[2], bounds[3],
                                                       width, height)

    im_zeros = np.zeros((img.shape[0], img.shape[1]))
    im_zeros = np.array(im_zeros, dtype=np.uint8)

    #print('RGBBBBBBBBBBBBBBBBBBBBBBB',  np.unique(img.reshape(-1, 3), axis=0))

    idx_red = np.where(np.all(img == (255,0,0), axis=-1))
    idx_green = np.where(np.all(img == (0,255,0), axis=-1))


    im_zeros[idx_red]=1
    im_zeros[idx_green]=1


    #save_tif_coregistered_with_img_id('./check/{}'.format(id), im_zeros*255, img1_id, img2_id, bounds, channels=1)

    all_polys = get_polygons_from_changes(im_zeros, bounds)

    print(len(all_polys))


    keep_polys = []


    for poly in all_polys:
        #print(poly)
        polyP = Polygon(poly['coordinates'][0])
#        bool_out = corridor.intersects(polyP)
        bool_out = polyP.intersects(corridor)

        if bool_out==True:
            keep_polys.append(polyP)
    #print('len', len(keep_polys))

    if keep_polys:
        final_polys = rasterio.features.rasterize(shapes=keep_polys, out_shape=(img.shape[0], img.shape[1]), all_touched=True, transform=geotiff_transform)
        print('UNIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII', np.unique(final_polys))
        idx0 = np.where(final_polys==0)   
        for ch in range(0, img.shape[2]):
            img[:,:,ch][idx0] = 0
        save_tif_coregistered_with_params('./PIPELINE_RESULTS/OUTPUT_incorridor/{}'.format(id), img, xparams, bounds, channels=3)

#        print(final_polys.shape) 
#        print('UNIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII', np.unique(final_polys)) 
#        print(np.unique(im_zeros))
    else:
        save_tif_coregistered_with_params('./PIPELINE_RESULTS/OUTPUT_incorridor/{}'.format(id), np.zeros((img.shape[0],img.shape[1], img.shape[2])), xparams, bounds, channels=3)




