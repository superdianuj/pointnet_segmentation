import os
import re
from glob import glob
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import open3d as o3
from point_net import PointNetSegHead
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from s3dis_dataset import S3DIS
import warnings
import torch.optim as optim
from point_net_loss import PointNetSegLoss
import math
import argparse
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
BATCH_SIZE = 16

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

# unique color map generated via
# https://mokole.com/palette.html
COLOR_MAP = {
    0  : (47, 79, 79),    # ceiling - darkslategray
    1  : (139, 69, 19),   # floor - saddlebrown
    2  : (34, 139, 34),   # wall - forestgreen
    3  : (75, 0, 130),    # beam - indigo
    4  : (255, 0, 0),     # column - red
    5  : (255, 255, 0),   # window - yellow
    6  : (0, 255, 0),     # door - lime
    7  : (0, 255, 255),   # table - aqua
    8  : (0, 0, 255),     # chair - blue
    9  : (255, 0, 255),   # sofa - fuchsia
    10 : (238, 232, 170), # bookcase - palegoldenrod
    11 : (100, 149, 237), # board - cornflower
    12 : (255, 105, 180), # stairs - hotpink
    13 : (0, 0, 0)        # clutter - black
}

v_map_colors = np.vectorize(lambda x : COLOR_MAP[x])

NUM_CLASSES = len(CATEGORIES)



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


EPOCHS = 0
LR = 0.0001

# use inverse class weighting
# alpha = 1 / class_bins
# alpha = (alpha/alpha.max())

# manually set alpha weights
alpha = np.ones(len(CATEGORIES))
alpha[0:3] *= 0.25 # balance background classes
alpha[-1] *= 0.75  # balance clutter class

gamma = 1



dir_models=os.getcwd()
mod_names=os.listdir('trained_models')
# sort
mod_names.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
model_names=[os.path.join(dir_models,model_name) for model_name in mod_names]
cand_model_name=model_names[-1]


file_path=args.path


if os.path.exists('unseen_data_predictions'):
    os.remove('unseen_data_predictions')

os.mkdir('unseen_data_predictions')


with open(file_path,'r') as file:
    for _ in range(12):
        file.readline()

    points=[]
    colors=[]

    for line in file:
        parts=line.strip().split()
        if len(parts)==7:
            x,y,z,intensity,r,g,b=map(float,parts)
            points.append([x,y,z])
            colors.append([r/255.0,g/255.0,b/255.0])

points,colors=np.array(points),np.array(colors)
pcd=o3.geometry.PointCloud()
pcd.points=o3.utility.Vector3dVector(points)
pcd.colors=o3.utility.Vector3dVector(colors)
o3.io.write_point_cloud('unseen_data_predictions/reference_scan.ply', pcd)

points=torch.tensor(points)
second_dim=15000*3
NUM_TEST_POINTS=15000*3
model=PointNetSegHead(num_points=NUM_TEST_POINTS,m=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(cand_model_name))
model.eval()
bs=(len(points)//second_dim)
Ps=[]
for i in range(bs):
    if i<bs:
        Ps.append(points[i*second_dim:(i+1)*second_dim])
Ps=torch.stack(Ps).float()
# Normalize each partitioned Point Cloud to (0, 1)
norm_points = Ps.clone()
norm_points = norm_points - norm_points.min(axis=1)[0].unsqueeze(1)
norm_points /= norm_points.max(axis=1)[0].unsqueeze(1)

p_choice=[]
mini_bs=32
with torch.no_grad():
    for i in range(bs//mini_bs+1):
        print(i)
        if i<bs//mini_bs:
            n_points=norm_points[i*mini_bs:(i+1)*mini_bs]
        else:
            n_points=norm_points[i*mini_bs:]
        # prepare data
        n_points = n_points.transpose(2, 1)

        preds, _, _ = model(n_points.cuda())

        # get metrics
        pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
        p_choice.append(pred_choice)
        torch.cuda.empty_cache()

pred_choice=torch.cat(p_choice)
colors=(np.vstack(v_map_colors(pred_choice.reshape(-1).to('cpu'))).T)/255
ps=points[:Ps.shape[0]*Ps.shape[1]]
pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(ps)
pcd.colors = o3.utility.Vector3dVector(colors)
o3.io.write_point_cloud('unseen_data_predictions/predicted_on_scan.ply', pcd)



