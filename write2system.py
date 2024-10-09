import numpy as np
from PIL import Image
import os
import rasterio
import rasterio.features
import pandas as pd
import psycopg
import datetime
import psycopg
import json
from tqdm import tqdm
from shapely.geometry import Polygon, box



def insert_items(cur, model_version, cycle_start, cycle_end, project_id, model_geometry, model_class, insert_time, detection_date, reference_date, detection_image_id, reference_image_id):
#    with psycopg.connect("dbname=sarccd2 host=devdb.orbitaleye.nl password=sarccd-db port=5433 user=postgres") as conn:
#        with conn.cursor() as cur:
    #print(model_version, cycle_start, cycle_end, project_id, model_geometry, model_class, insert_time, detection_date, reference_date, detection_image_id, reference_image_id)
    cur.execute(
        """
                insert into public.class_location_observations
                (
                model_version,
                cycle_start,
                cycle_end,
                project_id,
                model_geometry,
                model_class,
                insert_time,
                detection_date,
                reference_date,
                detection_image_id,
                reference_image_id
                ) values (%s, %s, %s, %s, ST_GeomFromGeoJSON(%s), %s, %s, %s, %s, %s, %s)
                """,
        (
                model_version,
                cycle_start,
                cycle_end,
                project_id,
                json.dumps(model_geometry),
                model_class,
                insert_time,
                detection_date,
                reference_date,
                detection_image_id,
                reference_image_id,
        ),
    )


def get_polygons_from_changes(changes, img_transform, connectivity=4):

#    height, width = changes.shape[0], changes.shape[1]
#    factor=1
#    geotiff_transform = rasterio.transform.from_bounds(poly[0], poly[1],
#                                                       poly[2], poly[3],
#                                                       width/factor, height/factor)


    result = []
    for shape, value in rasterio.features.shapes(changes, connectivity=connectivity, transform=img_transform):
        if value == 1:
            result.append(shape)

    return result


def extract_polygons_points(idx, img, pixel_transform, geotiff_transform):
    im_zeros = np.zeros((img.shape[0], img.shape[1]))
    im_zeros[idx]=1
    im_zeros = np.array(im_zeros, dtype=np.uint8)    
    #out = get_polygons_from_changes(im_zeros, img_bounds)
    out = get_polygons_from_changes(im_zeros[::-1,:], pixel_transform)
    #print(out)

    poly_list = []


    for o in out:
        print('OOOOOOOOoooooooooooooooo')
        print(o)
        pol_dict = {}
        pol_o = Polygon(o['coordinates'][0])
        simplified_polygon = pol_o.simplify(tolerance=2, preserve_topology=True)
        print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
        print(simplified_polygon)
        geo_simplified_polygon = ([geotiff_transform * (x, y) for x, y in simplified_polygon.exterior.coords])
#        print(geo_simplified_polygon)
        pol_dict['type'] = 'Polygon'
        pol_dict['coordinates'] = [geo_simplified_polygon]
        poly_list.append(pol_dict)


    return poly_list


ids = os.listdir('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS_arxeio/OUTPUT_reg/') ##here we put the regularized output!!!

colour_ids = [(0,   0,   0), (0, 255,   0), (255,   0,   0)]  #black, green, red
#/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/B or A
#ids = ['output_data_region_5996496.tif', 'output_data_region_5996665.tif']
model_version = 'b_mamba_v0'
cycle_start = '2022-11-01'
cycle_end = '2024-05-31'
project_id = 54

insert_time = datetime.datetime.now() 

with psycopg.connect("dbname=sarccd2 host=devdb.orbitaleye.nl password=sarccd-db port=5433 user=postgres") as conn:
    with conn.cursor() as cur:

        for _,id in enumerate(tqdm(ids)):
            im = rasterio.open('/home/mariapap/CODE/CLv2_eval/PIPELINE_RESULTS_arxeio/OUTPUT_reg/{}'.format(id))
            im_bounds = im.bounds
            #print(im.tags())

            detection_date = im.tags()['ctime_after']
            reference_date = im.tags()['ctime_before']
            detection_image_id = im.tags()['id_after']
            reference_image_id = im.tags()['id_before']

            im = im.read()
            #print(im.shape)
            im = np.transpose(im, (1,2,0))

            height, width = im.shape[0], im.shape[1]

            pixel_bounds = box(0, 0, width, height).bounds
            pixel_transform = rasterio.transform.from_bounds(pixel_bounds[0], pixel_bounds[1],
                                                                pixel_bounds[2], pixel_bounds[3],
                                                                width, height)


            geotiff_transform = rasterio.transform.from_bounds(im_bounds[0], im_bounds[1],
                                                                im_bounds[2], im_bounds[3],
                                                                width, height)




            #rn = [i for i in range(len(id)) if id.startswith('_', i)]
            #region_id = id[rn[-1]+1:-4]
            #print(region_id)
################################################################################################################################################
            #p_file_A = pd.read_pickle('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/A/' + 'data_region_{}.p'.format(region_id))
            #p_file_B = pd.read_pickle('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/B/' + 'data_region_{}.p'.format(region_id))

################################################################################################################################################

            idx_green= np.where(np.all(im == (0,255,0), axis=-1))  #red for removed buildings
            polys_green = extract_polygons_points(idx_green, im, pixel_transform, geotiff_transform)

            for pol in polys_green:
                #print('OOOOOOOOOOOOOOOOOOOOOOOOOOO', pol)

                insert_items(cur, model_version, cycle_start, cycle_end, project_id, pol, 'new', datetime.datetime.now(), detection_date, reference_date, detection_image_id, reference_image_id)



#            im_zeros = Image.fromarray(im_zeros)
#            im_zeros.save(id[:-4] + '.png')

            idx_red= np.where(np.all(im == (255,0,0), axis=-1))  #red for removed buildings
            polys_red = extract_polygons_points(idx_red, im, pixel_transform, geotiff_transform)
            for pol in polys_red:
                insert_items(cur, model_version, cycle_start, cycle_end, project_id, pol, 'removed', datetime.datetime.now(), detection_date, reference_date, detection_image_id, reference_image_id)
#            cur.connection.commit()

