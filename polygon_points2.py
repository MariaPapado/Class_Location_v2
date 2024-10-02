import rasterio
import numpy as np
import os
from PIL import Image
import rasterio.features as features
from shapely import geometry
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
import cv2
from affine import Affine
from shapely.affinity import scale
from datetime import datetime
import json


def get_polygons_from_changes(changes, img_transform, connectivity=4):

    result = []
    for shape, value in rasterio.features.shapes(changes, connectivity=connectivity, transform=img_transform):
        if value == 1:
            result.append(shape)

    return result



def save_tif_coregistered_with_img_id(filename, image, img_id_before, img_id_after, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly[0], poly[1],
                                                       poly[2], poly[3],
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

        dst.update_tags(img_id_before='{}'.format(img_id_before))
        dst.update_tags(img_id_after='{}'.format(img_id_after))
        
    dst.close()

    return True

def extract_polygons_points(idx, img, pixel_transform):
    im_zeros = np.zeros((img.shape[0], img.shape[1]))

    im_zeros[idx]=1
    im_zeros = np.array(im_zeros, dtype=np.uint8)    
    #out = get_polygons_from_changes(im_zeros, img_bounds)
    out = get_polygons_from_changes(im_zeros[::-1,:], pixel_transform)

    poly_list = []


    for o in out:
        pol_o = Polygon(o['coordinates'][0])

    #    print('POOOOOOOOOOOOOOOOOOOOOOOOOOL')
    #    print(pol_o)

        simplified_polygon = pol_o.simplify(tolerance=2, preserve_topology=True)
    #    print('SSSSSSSSSSSSSSSSSSSSSS')
    #    print(simplified_polygon)

        geo_simplified_polygon = Polygon([geotiff_transform * (x, y) for x, y in simplified_polygon.exterior.coords])


        poly_list.append(geo_simplified_polygon)

    cnt=1
    pol_values = []
    for pol in poly_list:
        pol_values.append((pol,1))
        cnt = cnt + 1


    burned = rasterio.features.rasterize(shapes=pol_values, out_shape=(img.shape[0], img.shape[1]), all_touched=True, transform=geotiff_transform) 

    return burned, poly_list


def save_to_geojson(filename_out, changes):
    output_geosjon = {'type': 'FeatureCollection', 'name': '{:%Y-%m-%d %H:%M:%S}'.format(datetime.today()),
                  'csr': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
                  'features': []}

    for i, change in enumerate(changes):
        tmp_i = {'type': 'Feature', 'properties': {'counter':i},
                'geometry': geometry.mapping(change)}
        output_geosjon['features'].append(tmp_i)
        
    json.dump(output_geosjon, open(filename_out, 'w'))
    


#img_ = rasterio.open('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS_withclouds/OUTPUT/output_region_5997149.tif')
#img_ = rasterio.open('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS_withclouds/OUTPUT_reg/5996980.tif')
img = rasterio.open('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg/5996659.tif')
img_bounds = img.bounds #BoundingBox(left=-78.76, bottom=43.11, right=-78.75, top=43.116)
img1_id = img.tags()['img_id_before']
img2_id = img.tags()['img_id_after']


img = img.read().transpose(1,2,0)

idx_green= np.where(np.all(img == (0,255,0), axis=-1))  #redgreen for removed buildings
idx_red= np.where(np.all(img == (255,0,0), axis=-1))  #redgreen for removed buildings

#img = cv2.imread('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg/5996659.tif', cv2.IMREAD_UNCHANGED)
height, width = img.shape[0], img.shape[1]

pixel_bounds = box(0, 0, width, height).bounds
pixel_transform = rasterio.transform.from_bounds(pixel_bounds[0], pixel_bounds[1],
                                                    pixel_bounds[2], pixel_bounds[3],
                                                    width, height)


geotiff_transform = rasterio.transform.from_bounds(img_bounds[0], img_bounds[1],
                                                    img_bounds[2], img_bounds[3],
                                                    width, height)



burned_appearing, pol_list = extract_polygons_points(idx_green, img, pixel_transform)
save_to_geojson('check.geojson', pol_list)

burned_disappearing, pol_list = extract_polygons_points(idx_red, img, pixel_transform)
save_to_geojson('check_dis.geojson', pol_list)



save_tif_coregistered_with_img_id('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg_points/5996659.tif', burned_appearing*255, img1_id, img2_id, img_bounds, channels=1)
save_tif_coregistered_with_img_id('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg_points/5996659_dis.tif', burned_disappearing*255, img1_id, img2_id, img_bounds, channels=1)

#burned = Image.fromarray(burned*255)
#burned.save('check.png')
#save_tif_coregistered('check.tif', burned*255, geotiff_transform, channels=1)
