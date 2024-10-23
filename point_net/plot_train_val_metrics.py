import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

train_acc=torch.load('score_data/train_accuracy.pth')
train_mcc=torch.load('score_data/train_mcc.pth')
train_iou=torch.load('score_data/train_iou.pth')

val_acc=torch.load('score_data/valid_accuracy.pth')
val_mcc=torch.load('score_data/valid_mcc.pth')
val_iou=torch.load('score_data/valid_iou.pth')



epochs=list(np.arange(0,50*(len(train_acc)),50))
plt.figure(figsize=(12, 6), dpi=180)
plt.plot(epochs,train_acc,label='Train Accuracy')
plt.plot(epochs,train_mcc,label='Train MCC')
plt.plot(epochs,train_iou,label='Train IOU')
plt.xlabel('Mini Batch Epcoshs')
plt.ylabel('Metrics')
plt.title('Training Metric vs Mini-Batch Epochs')
plt.legend()
plt.savefig('metrics.png')


plt.figure(figsize=(12, 6), dpi=180)
plt.plot(epochs,val_acc,label='Validation Accuracy')
plt.plot(epochs,val_mcc,label='Validation MCC')
plt.plot(epochs,val_iou,label='Validation IOU')
plt.xlabel('Mini Batch Epcoshs')
plt.ylabel('Metrics')
plt.title('Validation Metrics vs Mini-Batch Epochs')
plt.legend()
plt.savefig('metrics_val.png')
