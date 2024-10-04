import numpy as np
from PIL import Image
import os
import rasterio
import rasterio.features
import pandas as pd
import datetime
import json
from tqdm import tqdm
import json
from shapely import geometry


def save_tif_coregistered_with_params(filename, image, xparams, poly, channels=1, factor=1):
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

        dst.update_tags(**xparams)
#        dst.update_tags(img_id_after='{}'.format(xparams['id_before']))
        
    dst.close()

    return True

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


ids = os.listdir('./PIPELINE_RESULTS/OUTPUT_incorridor/')
#ids = ['output_data_region_5999346.tif']

colour_ids = [(0,   0,   0), (0, 255,   0), (255,   0,   0)]  #black, green, red
#/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/B or A
#ids = ['output_data_region_5996496.tif', 'output_data_region_5996665.tif']


with open('National-Fuel-2024_buildings.geojson') as f:
    sbuilds = json.load(f)


#print(sbuilds['features'][0])
#print('aaa')
#print(sbuilds['features'][1])
#print(sbuilds['features'][0]['geometry']['coordinates'])

# Create MultiPolygon of alll known building footprints
all_buildings = []
for building in sbuilds['features']:
    poly = geometry.shape(building['geometry'])
    if poly.type == 'Polygon':
        all_buildings.append(poly)
    else:
        for poly_i in poly.geoms:
            all_buildings.append(poly_i)

multipoly_buildings = geometry.MultiPolygon(all_buildings).buffer(0.)

all_polys = []
xparams={}
for _,id in enumerate(tqdm(ids)):
    im = rasterio.open('./PIPELINE_RESULTS/OUTPUT_incorridor/{}'.format(id))
    bounds = im.bounds

    xparams['id_before'] = im.tags()['id_before']
    xparams['id_after'] = im.tags()['id_after']

    xparams['ctime_before'] = im.tags()['ctime_before']
    xparams['ctime_after'] = im.tags()['ctime_after']

    im = im.read()
    #print(im.shape)
    im = np.transpose(im, (1,2,0))
    rn = [i for i in range(len(id)) if id.startswith('_', i)]
    region_id = id[rn[-1]+1:-4]
    print(region_id)
#    p_file_A = pd.read_pickle('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/A/' + 'data_region_{}.p'.format(region_id))
   # p_file_B = pd.read_pickle('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/B/' + 'data_region_{}.p'.format(region_id))

    im_zeros = np.zeros((im.shape[0], im.shape[1]))

    idx_green = np.where(np.all(im == (0,255,0), axis=-1))

    im_zeros[idx_green]=1
    im_zeros = np.array(im_zeros, dtype=np.uint8)


##########################################################################################################################################

    height, width = im.shape[0], im.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(bounds[0], bounds[1],
                                                       bounds[2], bounds[3],
                                                       width, height)
##########################################################################################################################################

    out = get_polygons_from_changes(im_zeros, bounds)
#    print(out)
    if out:
        del_polys = []
        for opol in out:
            opol = geometry.Polygon(opol['coordinates'][0])
    #        opol.intersects(multipoly_buildings)
            if opol.centroid.within(multipoly_buildings):
    #            print('True')
                del_polys.append(opol)
        if del_polys:
            burned = rasterio.features.rasterize(shapes=del_polys, out_shape=(im.shape[0], im.shape[1]), all_touched=True, transform=geotiff_transform) 
            idx_black = np.where(burned==1)
            im[idx_black] = [0,0,0]
    #print('uni', np.unique(burned))
    #burned[idx_black]=255
    #print('bbb', burned.shape)
    im = np.array(im, dtype=np.uint8)
    save_tif_coregistered_with_params('./PIPELINE_RESULTS/OUTPUT_comparegeojson/{}'.format(id), im, xparams, bounds, channels=3)

    im = Image.fromarray(im)


    
