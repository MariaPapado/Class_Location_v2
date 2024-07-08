import json
from shapely import geometry
from pimsys.regions.RegionsDb import RegionsDb
import rasterio.features
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from geopy.distance import geodesic
import requests
import cv2
from PIL import Image
from tqdm import tqdm


def get_utc_timestamp(x: datetime):
        return int((x - datetime(1970, 1, 1)).total_seconds())


def get_images(database, config, region, interval_utc):
    coords = [region['bounds'].centroid.x, region['bounds'].centroid.y]
    images = database.get_optical_images_containing_point_in_period(coords, interval_utc)

    wms_images = sorted(images, key=lambda x: x["capture_timestamp"])
    wms_images = [x for x in wms_images if x["source"] != "Sentinel-2"]
    wms_images = [x for x in wms_images if x['source'] == 'SkySat']

    all_image_ids = []
    all_images = []

    for image in wms_images:
        if image["wms_layer_name"] not in all_image_ids:
            all_image_ids.append(image["wms_layer_name"])
            all_images.append(image)
    # Sort images based on classification
#    return all_images
    return all_images

def get_image_from_layer(layer, region_bounds):
    layer_name = layer['wms_layer_name']

    # Define layer name
    pixel_resolution_x = layer["pixel_resolution_x"]
    pixel_resolution_y = layer["pixel_resolution_y"]

    region_width = geodesic((region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[1], region_bounds.bounds[2])).meters
    region_height = geodesic((region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[3], region_bounds.bounds[0])).meters

    width = int(round(region_width / pixel_resolution_x))
    height = int(round(region_height / pixel_resolution_y))

    arguments = {
        'layer_name': layer_name,
        'bbox': '%s,%s,%s,%s' % (region_bounds.bounds[0], region_bounds.bounds[1], region_bounds.bounds[2], region_bounds.bounds[3]),
        'width': width,
        'height': height
    }

    # get URL
    if 'image_url' in layer.keys():
        if layer['downloader'] == 'geoserve':
            arguments['bbox'] = '%s,%s,%s,%s' % (region_bounds[1], region_bounds[0], region_bounds[3], region_bounds[2])
            url = layer['image_url'] + "&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&CRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments
        elif layer['downloader'] == 'sentinelhub':
            url = layer['image_url']
        else:
            url = layer['image_url'] + "&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments
    else:
        url = "https://maps.orbitaleye.nl/mapserver/?map=/maps/_%(layer_name)s.map&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments

    if layer['downloader'] == 'geoserve':
        resp = requests.get(url, auth=('ptt', 'yOju6YLPK6Pnqm2C'))
    else:
        resp = requests.get(url, auth=('mapserver_user', 'tL3n3uoPnD8QYiph'))
	
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]

    return image


#file = 'National-Fuel-2024_buildings_merged.geojson'
file = './merged_geojson_masks/National-Grid-2024_buildings_merged.geojson'

# Loag geosjon
with open(file) as f:
    data = json.load(f)['features']

# Extract coordinates
polies = []
for poly in data:
    polies.append(geometry.shape(poly['geometry']).buffer(0.))

config = {
    "regions_db": {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
        }
    }

#customer = 'National-Fuel-2024'
customer = 'National-Grid-2024'

database = RegionsDb(config['regions_db'])
database_customer = database.get_regions_by_customer(customer)
#database.close()


dates = [datetime(2024, 1, 10), datetime(2024, 5, 6)]
interval_utc = [get_utc_timestamp(dates[0]), get_utc_timestamp(dates[1])]


# Select a region
for _, region in enumerate(tqdm(database_customer)):
    #print(region)
    #region = database_customer[74]

    images = get_images(database, config, region, interval_utc)

#    print('len', len(images))
    if images:
        img = get_image_from_layer(images[-1], region['bounds'])

        #print(img.shape)
        img_pil = Image.fromarray(img)


# get all intersecting buildings
        buildings = []
        for poly in polies:
    #	if poly.type == 'Point':
    #		continue
            if poly.intersects(region['bounds']):
                buildings.append(poly)  # poly.intersection(region['bounds']))

        all_builings = geometry.MultiPolygon(buildings).buffer(0.)


        if all_builings:
#            print('type', type(all_builings))
            zero_image = np.zeros(([3, img.shape[0], img.shape[1]]))

# Convert building to a mask
            geotiff_transform = rasterio.transform.from_bounds(region['bounds'].bounds[0], region['bounds'].bounds[1],
                                                   region['bounds'].bounds[2], region['bounds'].bounds[3],
                                                   zero_image.shape[2], zero_image.shape[1])

            mask = rasterio.features.geometry_mask([all_builings], [zero_image.shape[1], zero_image.shape[2]], transform=geotiff_transform, all_touched=True, invert=True)
            mask = mask.astype(np.uint8)*255
            #print('type', np.unique(mask))


            mask = Image.fromarray(mask)
            mask.save('./img_n_labels/{}_{}_mask.png'.format(region['id'], region['region_customer_id']))
            img_pil.save('./img_n_labels/{}_{}_img.png'.format(region['id'], region['region_customer_id']))
