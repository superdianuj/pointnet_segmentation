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
# from open3d import JVisualizer # For Colab Visualization
# from open3d.web_visualizer import draw # for non Colab

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# draw(pcd)
from point_net import PointNetSegHead
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if not os.path.exists('s3dis_test_pointclouds'):
    os.rmdir('s3dis_test_pointclouds')

os.mkdir('s3dis_test_pointclouds')


# dataset
curr_dir=os.getcwd()
ROOT = os.path.join(curr_dir,'Stanford3dDataset_v1.2_Reduced_Partitioned_Aligned_Version')
# feature selection hyperparameters
# feature selection hyperparameters
NUM_TRAIN_POINTS = 4096 # train/valid points
NUM_TEST_POINTS = 15000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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


from torch.utils.data import DataLoader
from s3dis_dataset import S3DIS

# get datasets
s3dis_train = S3DIS(ROOT, area_nums='1-4', npoints=NUM_TRAIN_POINTS, r_prob=0.25)
s3dis_valid = S3DIS(ROOT, area_nums='5', npoints=NUM_TRAIN_POINTS, r_prob=0.)
s3dis_test = S3DIS(ROOT, area_nums='5', npoints=NUM_TEST_POINTS)

# get dataloaders
train_dataloader = DataLoader(s3dis_train, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(s3dis_valid, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(s3dis_test, batch_size=BATCH_SIZE, shuffle=False)

points, targets = s3dis_train[1]

points,targets=next(iter(train_dataloader))



def compute_iou(targets, predictions):

    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union


dir_models=os.getcwd()
dir_models=os.path.join(dir_models,'trained_models')
mod_names=os.listdir(dir_models)
# sort
mod_names.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
model_names=[os.path.join(dir_models,model_name) for model_name in mod_names]


acc_record=[]
mcc_record=[]
iou_record=[]

for model_name in model_names:
    name_of_model=model_name.split('/')[-1].split('.')[0]
    model=PointNetSegHead(num_points=NUM_TEST_POINTS,m=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)
    t_accuracy=0
    t_mcc=0
    t_iou=0
    batch_count=0
    total_points=[]
    total_labels=[]
    total_preds=[]
    for batch in test_dataloader:

        points, targets = batch
        # points,targets=points[0:1],targets[0:1]


        points = points.to(DEVICE)
        targets = targets.to(DEVICE)



        # Normalize each partitioned Point Cloud to (0, 1)
        norm_points = points.clone()
        norm_points = norm_points - norm_points.min(axis=1)[0].unsqueeze(1)
        norm_points /= norm_points.max(axis=1)[0].unsqueeze(1)

        with torch.no_grad():

            # prepare data
            norm_points = norm_points.transpose(2, 1)
            targets = targets.squeeze()

            # run inference
            preds, _, _ = model(norm_points)

            # get metrics
            pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

            correct = pred_choice.eq(targets.data).cpu().sum()
            accuracy = correct/float(points.shape[0]*NUM_TEST_POINTS)
            mcc = mcc_metric(preds.transpose(2, 1), targets)
            iou = compute_iou(targets, pred_choice)
            break


    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points.permute(2, 0, 1).reshape(3, -1).to('cpu').T)
    pcd.colors = o3.utility.Vector3dVector(np.vstack(v_map_colors(targets.reshape(-1).to('cpu'))).T/255)
    o3.io.write_point_cloud(f's3dis_test_pointclouds/ground_truth_{name_of_model}.ply',pcd)

    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points.permute(2, 0, 1).reshape(3, -1).to('cpu').T)
    pcd.colors = o3.utility.Vector3dVector(np.vstack(v_map_colors(pred_choice.reshape(-1).to('cpu'))).T/255)
    o3.io.write_point_cloud(f's3dis_test_pointclouds/full_predicted_{name_of_model}.ply', pcd)
    
    print(f'Accuracy: {accuracy:.4f} - MCC: {mcc:.4f} - IOU: {iou:.4f}')
    acc_record.append(accuracy.item())
    mcc_record.append(mcc.item())
    iou_record.append(iou.item())



epochs=list(np.arange(0,50*(len(acc_record)),50))
plt.figure(figsize=(12, 6), dpi=180)
plt.plot(epochs,acc_record, label='Accuracy')
plt.plot(epochs,mcc_record, label='MCC')
plt.plot(epochs,iou_record, label='IOU')
plt.xlabel('Epcoshs')
plt.ylabel('Metrics')
plt.title('Metrics vs Epochs')
plt.legend()
plt.savefig('test_s3dis_metrics.png')
