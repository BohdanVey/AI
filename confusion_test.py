import numpy as np
import torch
import sklearn.metrics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import torchvision
import seaborn as sn

def plot_to_tensorboard(writer, conf_mat, step):

    fig = plt.figure()
    sn.heatmap(conf_mat / conf_mat.max(), annot=True)

    writer.add_figure('confusion_matrix', fig,step)


train_writer = SummaryWriter(log_dir='train')

for i in range(10):
    x = torch.rand((4, 7, 4, 4))
    x = (x > 0.5) * 1
    x1 = torch.argmax(x, dim=1)
    y = torch.rand((4, 7, 4, 4))
    y1 = torch.argmax(y, dim=1)
    matrix = sklearn.metrics.confusion_matrix(x1.reshape(-1), y1.reshape(-1))

    plot_to_tensorboard(train_writer, matrix, i)
