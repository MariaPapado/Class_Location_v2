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


ids = os.listdir('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/PIPELINE_RESULTS_NAT/OUTPUT_filtered/') ##here we put the regularized output!!!

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
            im = rasterio.open('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/PIPELINE_RESULTS_NAT/OUTPUT_filtered/{}'.format(id))
            im = im.read()
            #print(im.shape)
            im = np.transpose(im, (1,2,0))
            rn = [i for i in range(len(id)) if id.startswith('_', i)]
            region_id = id[rn[-1]+1:-4]
            #print(region_id)
            p_file_A = pd.read_pickle('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/A/' + 'data_region_{}.p'.format(region_id))
            p_file_B = pd.read_pickle('/home/mariapap/CODE/ChangeOS/MAMBA/MambaCD/NatBasemaps/basemaps_natfuel_test/out/B/' + 'data_region_{}.p'.format(region_id))

            detection_date = p_file_B['capture_timestamp']
            reference_date = p_file_A['capture_timestamp']
            detection_image_id = p_file_B['image_id']
            reference_image_id = p_file_A['image_id']


            im_zeros = np.zeros((im.shape[0], im.shape[1]))
            idx= np.where(np.all(im == (255,0,0), axis=-1))  #red for removed buildings
            im_zeros[idx]=1
            im_zeros = np.array(im_zeros, dtype=np.uint8)    
            out = get_polygons_from_changes(im_zeros, p_file_A['poly'])
            model_geometry = out


            for pol in out:
                 insert_items(cur, model_version, cycle_start, cycle_end, project_id, pol, 'removed', datetime.datetime.now(), detection_date, reference_date, detection_image_id, reference_image_id)



#            im_zeros = Image.fromarray(im_zeros)
#            im_zeros.save(id[:-4] + '.png')

            im_zeros = np.zeros((im.shape[0], im.shape[1]))
            idx= np.where(np.all(im == (0,255,0), axis=-1))  #green for new buildings
            im_zeros[idx]=1
            im_zeros = np.array(im_zeros, dtype=np.uint8)    
            out = get_polygons_from_changes(im_zeros, p_file_A['poly'])
            for pol in out:
                 insert_items(cur, model_version, cycle_start, cycle_end, project_id, pol, 'new', datetime.datetime.now(), detection_date, reference_date, detection_image_id, reference_image_id)
#            cur.connection.commit()

