import rasterio
import numpy as np
import os
from PIL import Image
import rasterio.features as features
from shapely import geometry
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
import cv2

def get_polygons_from_changes(changes, poly, connectivity=4):

    height, width = changes.shape[0], changes.shape[1]
    factor=1
    geotiff_transform = rasterio.transform.from_bounds(poly[0], poly[1],
                                                       poly[2], poly[3],
                                                       width/factor, height/factor)


    result = []
    for shape, value in rasterio.features.shapes(changes, connectivity=connectivity, transform=geotiff_transform):
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

#img_ = rasterio.open('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS_withclouds/OUTPUT/output_region_5997149.tif')
#img_ = rasterio.open('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS_withclouds/OUTPUT_reg/5996980.tif')
img = rasterio.open('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg/5996659.tif')
img_bounds = img.bounds
img1_id = img.tags()['img_id_before']
img2_id = img.tags()['img_id_after']


img = img.read().transpose(1,2,0)
#img = cv2.imread('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg/5996659.tif', cv2.IMREAD_UNCHANGED)
height, width = img.shape[0], img.shape[1]

pixel_bounds = box(0, 0, width, height).bounds
pixel_transform = rasterio.transform.from_bounds(pixel_bounds[0], pixel_bounds[1],
                                                    pixel_bounds[2], pixel_bounds[3],
                                                    width, height)


geotiff_transform = rasterio.transform.from_bounds(img_bounds[0], img_bounds[1],
                                                    img_bounds[2], img_bounds[3],
                                                    width, height)

im_zeros = np.zeros((img.shape[0], img.shape[1]))
idx= np.where(np.all(img == (0,255,0), axis=-1))  #redgreen for removed buildings
im_zeros[idx]=1
im_zeros = np.array(im_zeros, dtype=np.uint8)    
#out = get_polygons_from_changes(im_zeros, img_bounds)
out = get_polygons_from_changes(im_zeros, pixel_bounds)


poly_list = []


for o in out:
    pol_o = Polygon(o['coordinates'][0])
    print('POOOOOOOOOOOOOOOOOOOOOOOOOOL')
    print(pol_o)

    simplified_polygon = pol_o.simplify(tolerance=2, preserve_topology=True)
    print('SSSSSSSSSSSSSSSSSSSSSS')
    print(simplified_polygon)

    geo_simplified_polygon = Polygon([geotiff_transform * (x, y) for x, y in simplified_polygon.exterior.coords])
    print('GEOOOOOOOO', geo_simplified_polygon)

    poly_list.append(geo_simplified_polygon)



cnt=1
pol_values = []
for pol in poly_list:
    pol_values.append((pol,1))
    cnt = cnt + 1


burned = rasterio.features.rasterize(shapes=pol_values, out_shape=(img.shape[0], img.shape[1]), all_touched=True, transform=geotiff_transform) 


print(burned.shape)
print(np.unique(burned))
#burned = Image.fromarray(burned*255)
#burned.save('check.png')

save_tif_coregistered_with_img_id('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS/OUTPUT_reg_points/5996659.tif', burned*255, img1_id, img2_id, img_bounds, channels=1)

#save_tif_coregistered('check.tif', burned*255, geotiff_transform, channels=1)
