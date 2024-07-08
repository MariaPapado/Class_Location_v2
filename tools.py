from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
import rasterio
import cv2
import numpy as np
import torch
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage.morphology import area_opening
from PIL import Image

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi



def filter_small_contours(im):
    im = np.array(im, dtype=np.uint8)
    threshold_area = 50     #threshold area 
    contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)   
    k_contours=[]

    for cnt in contours:    
#        print(cnt)    
        area = cv2.contourArea(cnt)         
#        print(area)
        if area > threshold_area:
            #Put your code in here
            k_contours.append(cnt)

    fim = np.zeros((im.shape[0], im.shape[1]))
    cv2.fillPoly(fim, pts =k_contours, color=(1,1,1))
#    print(fim.shape)
#    cv2.imwrite(id[:-4] + '__filter.png', fim)
    return fim


def check_shape_and_resize(raster_before, raster_after):
    h_before, w_before = raster_before.height, raster_before.width
    h_after, w_after = raster_after.height, raster_after.width
    img_before = raster_before.read()
    img_after = raster_after.read()
    img_before, img_after = np.transpose(img_before, (1,2,0)), np.transpose(img_after, (1,2,0))
    if (h_before + w_before) < (h_after + w_after):
        img_after = cv2.resize(img_after, (w_before, h_before), cv2.INTER_NEAREST)
        return img_before, img_after, raster_before.bounds  
    elif (h_before + w_before) > (h_after + w_after):    
        img_before = cv2.resize(img_before, (w_after, h_after), cv2.INTER_NEAREST)
        return img_before, img_after, raster_after.bounds 
    else:
        return img_before, img_after, raster_before.bounds




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


def register_image_pair(image_before, image_after):
    f_image_after = image_after.copy()
    image_before, image_after = np.array(image_before)/255.0, np.array(image_after)/255.0

    image_before = np.transpose(image_before, (2,0,1))
    image_after = np.transpose(image_after, (2,0,1))

    extractor = SuperPoint(max_num_keypoints=2048).eval()#.cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval()#.cuda()  # load the matcher

    feats1 = extractor.extract(torch.from_numpy(image_before).float())  # auto-resize the image, disable with resize=None
    feats0 = extractor.extract(torch.from_numpy(image_after).float())


    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    height, width = image_before.shape[1:]

    img2_color = np.moveaxis(image_after, 0, -1)

    height, width = image_before.shape[1:]

    homography, mask = cv2.estimateAffine2D(points0.numpy(), points1.numpy(), cv2.RANSAC)
    homography = homography.astype(np.float32)

############################check################################################
    if not (
        (np.abs(homography)[1][0] + np.abs(homography)[0][1] < 0.02)
        and (np.abs(homography)[0][0] + np.abs(homography)[1][1] < 2 + 0.01)
        and (np.abs(homography)[0][0] + np.abs(homography)[1][1] > 2 - 0.01)
        and len(points0) > 75
    ):
          print('WRONG!!!!!!!!!!!!!!!!!!!!!!!!')
          return f_image_after, False
          
############################check################################################


    transformed_img = cv2.warpAffine(img2_color, homography, (width, height))

    transformed_img = np.array(transformed_img*255, dtype=np.uint8)


    return transformed_img, True

def post_processing(buildings_before, buildings_after):
#    buildings_before = np.array(buildings_before, dtype=np.uint8)
#    buildings_after = np.array(buildings_after, dtype=np.uint8)

    diff = buildings_after - buildings_before
    diff[diff<0]=2


#    img_contour = np.zeros((buildings_before.shape[0], buildings_before.shape[1], 3))
#    intersection = np.logical_and(buildings_before, buildings_after)  

#    idx = np.where(intersection==True)
#    buildings_after[idx] = 0

#    new_c, hierarchy = cv2.findContours(buildings_after, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#    cv2.fillPoly(img_contour, new_c, (255, 255, 255))

#    img_contour = np.array(img_contour[:,:,0], dtype=np.uint8)

#    idx255 = np.where(img_contour==255)
#    img_contour[idx255] = 1

#####################################
    footprint = disk(4)
    out = erosion(diff, footprint)

    out = area_opening(out, area_threshold=64, connectivity=1)

    diff[out==0]=0
    return diff
 
def visualize(dam):
    dam = np.array(dam, dtype=np.uint8)
    dam = Image.fromarray(dam)
    dam.putpalette([0, 0, 0,
                    0, 255, 0,
                    255, 0, 0])
    dam = dam.convert('RGB')
    dam = np.asarray(dam)

    return dam
 

def apply_watershed(label_before, thresh):

    thresh = np.array(thresh, dtype=np.uint8)
    D = ndi.distance_transform_edt(thresh)
    coords = peak_local_max(D, footprint=np.ones((20, 20)), labels=thresh)
    mask = np.zeros(D.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-D, markers, mask=thresh)



#    D = ndi.distance_transform_edt(thresh)
#    localMax = peak_local_max(D,  min_distance=20, labels=thresh)

#    markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
#    labels = watershed(-D, markers, mask=thresh)


    #print(np.unique(labels))

    #cv2.imwrite('wat.png', labels*4)

#    print('nppppp', labels.shape, np.unique(labels))
    appears = np.zeros((labels.shape[0], labels.shape[1]))

    for label in np.unique(labels)[1:]:
        # if the label is zero, we are examining the 'background'
        # so simply ignore it  
        idx = np.where(labels==label)
        #print(idx[0].shape)
        before_builds = label_before[idx]
        builds = np.count_nonzero(before_builds==1)

        if (builds/len(before_builds))<0.02:
            appears[idx] = 1

    return appears
    #cv2.imwrite('appears.png', appears)
