__author__ = 'hcaesar'

# Helper functions used to convert between different formats for the
# COCO Stuff Segmentation Challenge.
#
# Note: Some functions use the Pillow image package, which may need
# to be installed manually.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

import numpy as np
from pycocotools import mask
from PIL import Image, ImagePalette  # For indexed images
import matplotlib  # For Matlab's color maps


def segmentationToCocoMask(label_map, label_id):
    '''
    Encodes a segmentation mask using the Mask API.
    :param label_map: [h x w] segmentation map that indicates the label of each pixel
    :param label_id: the label from labelMap that will be encoded
    :return: Rs - the encoded label mask for label 'labelId'
    '''
    label_mask = label_map == label_id
    label_mask = np.expand_dims(label_mask, axis=2)
    label_mask = label_mask.astype('uint8')
    label_mask = np.asfortranarray(label_mask)
    Rs = mask.encode(label_mask)
    assert len(Rs) == 1
    Rs = Rs[0]

    return Rs


def segmentationToCocoResult(labelMap, imgId, stuffStartId=92):
    '''
    Convert a segmentation map to COCO stuff segmentation result format.
    :param labelMap: [h x w] segmentation map that indicates the label of each pixel
    :param imgId: the id of the COCO image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Get stuff labels
    shape = labelMap.shape
    if len(shape) != 2:
        raise Exception(('Error: Image has %d instead of 2 channels! Most likely you '
                         'provided an RGB image instead of an indexed image (with or without color palette).') % len(shape))
    [h, w] = shape
    assert h > 0 and w > 0
    labelsAll = np.unique(labelMap)
    labelsStuff = [i for i in labelsAll if i >= stuffStartId]

    # Add stuff annotations
    anns = []
    for labelId in labelsStuff:
        # Create mask and encode it
        Rs = segmentationToCocoMask(labelMap, labelId)

        # Create annotation data and add it to the list
        anndata = {}
        anndata['image_id'] = int(imgId)
        anndata['category_id'] = int(labelId)
        anndata['segmentation'] = Rs
        anns.append(anndata)
    return anns


def cocoSegmentationToSegmentationMap(coco, img_id, cat_id, check_unique_pixel_label=True, include_crowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param img_id: the id of the COCO image
    :param cat_id: the category wanted
    :param check_unique_pixel_label: (optional) whether every pixel can have at most one label
    :param include_crowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h,w] segmentation map that indicates the label of each pixel
    '''

    # Init
    current_img = coco.imgs[img_id]
    image_size = (current_img['height'], current_img['width'])
    label_map = np.zeros(image_size)

    # Get annotations of the current image (may be empty)
    if include_crowd:
        ann_ids = coco.getAnnIds(imgIds=img_id)
    else:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    img_annotations = coco.loadAnns(ann_ids)

    # Combine all annotations of this image in labelMap
    for a in range(0, len(img_annotations)):
        if cat_id != img_annotations[a]['category_id']:
            continue

        true_false_label_mask = coco.annToMask(img_annotations[a]) == 1
        the_label_is_the_color = img_annotations[a]['category_id']

        if check_unique_pixel_label and (label_map[true_false_label_mask] != 0).any():
            raise Exception('Error: Some pixels have more than one label (image %d)!' % img_id)

        label_map[true_false_label_mask] = the_label_is_the_color

    assert label_map.shape[0] == image_size[0]
    assert label_map.shape[1] == image_size[1]
    assert 2 == len(label_map.shape)
    return label_map.astype(np.int8)


def pngToCocoResult(pngPath, imgId, stuffStartId=92):
    '''
    Reads an indexed .png file with a label map from disk and converts it to COCO result format.
    :param pngPath: the path of the .png file
    :param imgId: the COCO id of the image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Read indexed .png file from disk
    im = Image.open(pngPath)
    labelMap = np.array(im)

    # Convert label map to COCO result format
    anns = segmentationToCocoResult(labelMap, imgId, stuffStartId)
    return anns


def cocoSegmentationToPng(coco, img_id, cat_id, output_file, check_unique_pixel_label=True, include_crowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map and write it to disk.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param img_id: the COCO id of the image (last part of the file name)
    :param cat_id: the category wanted
    :param output_file: the path of the .png file
    :param check_unique_pixel_label: (optional) whether every pixel can have at most one label
    :param include_crowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: None
    '''

    # Create label map
    label_map = cocoSegmentationToSegmentationMap(coco, img_id, cat_id, check_unique_pixel_label=check_unique_pixel_label, include_crowd=include_crowd)

    # Get color map and convert to PIL's format
    cmap = get_cmap_for_categories()
    cmap = (cmap * 255).astype(int)
    padding = np.zeros((256 - cmap.shape[0], 3), np.int8)
    cmap = np.vstack((cmap, padding))
    cmap = cmap.reshape((-1))
    assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'

    # Write to png file
    png = Image.fromarray(label_map).convert('P')
    png.putpalette(list(cmap))
    png.save(output_file, format='PNG')


def get_cmap_for_categories():
    stuff_start_id = 1
    stuff_end_id = 90
    cmap_name = 'jet'

    # Get jet color map from Matlab
    label_count = stuff_end_id - stuff_start_id + 1
    cmap_gen = matplotlib.cm.get_cmap(cmap_name, label_count)
    cmap = cmap_gen(np.arange(label_count))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(label_count)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black color for 'unlabeled' class
    cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    return cmap


def getCMap(stuffStartId=92, stuffEndId=182, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addThings: (optional) whether to add a color for the 91 thing classes
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :param addOther: (optional) whether to add a color for the 'other' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black (or any other) color for each thing class
    if addThings:
        thingsPadding = np.zeros((stuffStartId - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    # Add yellow/orange color for 'other' class
    if addOther:
        cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap
