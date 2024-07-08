import os
import re
from glob import glob
import time
import numpy as np
import pandas as pd
import open3d as o3




curr_dir=os.cwd()
ROOT=os.path.join(curr_dir,'point_net/Stanford3dDataset_v1.2_Aligned_Version')
SAVE_PATH = os.path.join(curr_dir,'point_net/Stanford3dDataset_v1.2_Reduced_Aligned_Version')
PARTITION_SAVE_PATH = os.path.join(curr_dir,'point_net/Stanford3dDataset_v1.2_Reduced_Partitioned_Aligned_Version')

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

if not os.path.exists(PARTITION_SAVE_PATH):
    os.mkdir(PARTITION_SAVE_PATH)

CATEGORIES = {
    'ceiling'  : 0, 
    'floor'    : 1, 
    'wall'     : 2, 
    'beam'     : 3, 
    'column'   : 4, 
    'window'   : 5,
    'door'     : 6, 
    'table'    : 7, 
    'chair'    : 8, 
    'sofa'     : 9, 
    'bookcase' : 10, 
    'board'    : 11,
    'stairs'   : 12,
    'clutter'  : 13
}

CATEGORIES = {
    'ceiling'  : 0,
    'floor'    : 1,
    'wall'     : 2,
    'beam'     : 3,
    'column'   : 4,
    'window'   : 5,
    'door'     : 6,
    'table'    : 7,
    'chair'    : 8,
    'sofa'     : 9,
    'bookcase' : 10,
    'board'    : 11,
    'stairs'   : 12,
    'clutter'  : 13
}

area_nums = '1-6' # decide on the number of areas to obtain
area_dict = {}

# get areas based on split
areas = glob(os.path.join(ROOT, f'Area_[{area_nums}]*'))

for area in areas:
    # get all subfolders in area (corresponds to disjoint spaces (or locations))
    spaces = next(os.walk(area))[1]

    # get dict to store spaces
    space_dict = {}

    # for each space
    for space in spaces:
        space = os.path.join(area, space)
        annotations = os.path.join(space, 'Annotations')

        # get individual segmentation filepaths
        segments = glob(os.path.join(annotations, '*.txt'))

        # update space dict
        space_dict.update({space.split('/')[-1] : segments})

    # update area dict
    area_dict.update({area.split('/')[-1] : space_dict})


def get_space_data(space_segments, categories=CATEGORIES):
    ''' Obtains space data in (x,y,z),cat format all types are float32
        Inputs:
            space_segments - (list) filepaths to all annotaed space segments
                            for the current space.
                            e.g. area_dict['Area_1']['conferenceRoom_2']
            categories - (dict) maps string category to numeric category
        Outputs:
            space_data - (array)
        '''
    # space data list (x,y,z, cat)
    space_data = []
    for seg_path in space_segments:

        # get truth category and xyz points
        cat = CATEGORIES[seg_path.split('/')[-1].split('_')[0]]
        xyz = pd.read_csv(seg_path, header=None, sep=' ',
                          dtype=np.float32, usecols=[0,1,2]).to_numpy()

        # add truth to xyz points and add to space list
        space_data.append(np.hstack((xyz,
                                     np.tile(cat, (len(xyz), 1)) \
                                     .astype(np.float32))))

    # combine into single array and return
    return np.vstack(space_data)


tic = time.time()

for area in area_dict:
    # create new directory
    save_dir = os.path.join(SAVE_PATH, area)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for space in area_dict[area]:
        # obtain xyz points with truth labels
        space_data = pd.DataFrame(get_space_data(area_dict[area][space]))

        # save as .hdf5 file in new directory
        save_path = os.path.join(save_dir, space + '.hdf5')
        space_data.to_hdf(save_path, key='space_data')


toc = time.time()
print(toc - tic)

# space_data = pd.read_hdf(os.path.join(save_dir, space + '.hdf5'), key='space_data').to_numpy()
space_data = get_space_data(area_dict['Area_3']['conferenceRoom_1'], categories=CATEGORIES)
def get_slice(points, xyz_s, xpart, ypart):
    ''' Obtains Point Cloud Slices from the (x,y) partitions
        By default this will obtain roughly 1x1 partitions
        inputs:
            points - (array) could be xyz, rgb or any input array
            xyz_s - (Nx3 array) 0 min shifter point cloud array
            xpart - xpartitions [[lower, upper]]
            ypart - ypartitions [[lower, upper]]
        '''
    x_slice = (xyz_s[:, 0] >= xpart[0]) \
              & (xyz_s[:, 0] <= xpart[1])

    y_slice = (xyz_s[:, 1] >= ypart[0]) \
              & (xyz_s[:, 1] <= ypart[1])

    return points[x_slice & y_slice, :]


def get_partitions(xyz, xyz_s, c=1.):
    ''' Obtains Point Cloud Space Partitions
        Inputs:
            xyz_s - (Nx3 array) 0 min shifted point cloud array
            c - (float) factor for deciding how many partitions to create (larger --> less partitions)
        Outputs:
            partitions - (tuple) x and y parition arrays with
                         format: [[lower, upper]]
        '''
    ## get number of x, y bins
    range_ = np.abs(xyz.max(axis=0) - xyz.min(axis=0))
    num_xbins, num_ybins, _ = np.uint8(np.round(range_ / c))

    # uncomment this to generate ~1x1m partitions
    # num_xbins, num_ybins, _ = np.uint8(np.ceil(np.max(xyz_s, 0)))

    ## get x, y bins
    _, xbins = np.histogram(xyz_s[:, 0], bins=num_xbins)
    _, ybins = np.histogram(xyz_s[:, 1], bins=num_ybins)

    ## get x y space paritions
    x_parts = np.vstack((xbins[:-1], xbins[1:])).T
    y_parts = np.vstack((ybins[:-1], ybins[1:])).T

    return x_parts, y_parts


tic = time.time()

num_invalid_partitions = 0

for area in area_dict:
    # create new directory
    save_dir = os.path.join(PARTITION_SAVE_PATH, area)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for space in area_dict[area]:
        # obtain xyz points with truth labels
        space_data = get_space_data(area_dict[area][space])

        # obtain x, y partitions
        xyz = space_data[:, :3]

        # get 0 min shifted points
        xyz_s = xyz - xyz.min(axis=0)
        x_parts, y_parts = get_partitions(xyz, xyz_s, c=1.5)

        # counter for parition saving
        i = 0
        for x_part in x_parts:
            for y_part in y_parts:

                space_slice = pd.DataFrame(get_slice(space_data, xyz_s, x_part, y_part))

                # only save if partition has at least 100 points:
                if len(space_slice) > 100:
                    i += 1
                    save_path = os.path.join(save_dir, space + f'_partition{i}_.hdf5')
                    space_slice.to_hdf(save_path, key='space_slice')
                else:
                    num_invalid_partitions += 1


toc = time.time()
print(toc - tic)


