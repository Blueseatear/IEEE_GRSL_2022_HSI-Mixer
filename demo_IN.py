# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:01:34 2022

@author: Blues
"""
import os
import sys

import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import collections
import logging
import argparse
from Utils import utils
from torch.utils import data
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn import metrics, preprocessing
from Utils.utils import cal_results, predVisIN
from Utils.Load_data import sampling, zeroPadding_3D, Getting_HSIsets, HSIDataset, Hybrid_HSIDataset, indexToAssignment, selectNeighboringPatch, all_HSI_Data_pred
import collections
from torchscan import summary
from models.hsi_mixer import HSI_Mixer_Net
#from Utils.hybrid_measurement import spectral_matrixs

parser = argparse.ArgumentParser("IN")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='IN', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--depth', type=int, default=1, help='Layer Depth')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument("--dataset", type=str, default="IN", help='dataset for experimental analysis')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--model', type=str, default='HSI_Mixer_Net', help='select network to train')
parser.add_argument('--phi', type=str, default='hybrid', help='sequential order of network')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
# parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

#Fix Random seed = 2
np.random.seed(args.seed)
cudnn.benchmark = True
torch.manual_seed(args.seed)

cudnn.enabled=True
torch.cuda.manual_seed(args.seed)



args.save = '{}-train-seed{}-model-{}-arch-{}-{}-lr{}'.format(args.set, args.seed ,args.model, args.phi, time.strftime("%Y%m%d-%H%M%S"), args.learning_rate)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('train_IN_hsimixer.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == "IN":
    data_IN = sio.loadmat(args.data+"/Indian_pines_corrected.mat")["indian_pines_corrected"]
    gt_IN = sio.loadmat(args.data+"/Indian_pines_gt.mat")["indian_pines_gt"]
    
elif args.dataset == "UP":
    data_IN = sio.loadmat(args.data+"/PaviaU.mat")["paviaU"]
    gt_IN = sio.loadmat(args.data+"/PaviaU_gt.mat")["paviaU_gt"]
    #VAL_SIZE = 4281
    #used_samples = [75,220,24,35,18,59,16,43,10]
    
elif args.dataset == "KSC":
    data_IN = sio.loadmat(args.data+"/KSC.mat")["KSC"]
    gt_IN = sio.loadmat(args.data+"/KSC_gt.mat")["KSC_gt"]
    #used_samples = [36,11,12,12,10,11,6,20,24,19,21,24,44]
    
elif args.dataset == "Salinas":
    data_IN = sio.loadmat(args.data+"/Salinas_corrected.mat")["salinas_corrected"]
    gt_IN = sio.loadmat(args.data+"/Salinas_gt.mat")["salinas_gt"]
    
elif args.dataset == "Houston":
    data_IN = sio.loadmat(args.data+"/houston15.mat")['data']
    mask_train = sio.loadmat(args.data+"/houston15_mask_train.mat")["mask_train"]
    mask_test = sio.loadmat(args.data+"/houston15_mask_test.mat")["mask_test"]
    
print("Load Raw Data Successfully! Data Shape", data_IN.shape)

# Input dataset configuration to generate 103x7x7 HSI samples
new_gt_IN = gt_IN

# preprocess data 
#data_IN = preprocessing.scale(data_IN)

# channel-wise normalize input data
#data -= np.sum(data)

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)

MAX = data_IN.max()
data_IN = np.transpose(data_IN, (2,0,1))

data_IN = data_IN - np.mean(data_IN, axis=(1,2), keepdims=True)
data_IN = data_IN / MAX

data_hsi = data_IN.reshape(np.prod(data_IN.shape[:1]),np.prod(data_IN.shape[1:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)

nb_classes = max(gt)
INPUT_DIMENSION_CONV = data_hsi.shape[0]
PATCH_LENGTH = 4 

whole_data = data_hsi.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])
padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

sample_200 = [2, 27, 19, 4, 9, 14, 2, 10, 3, 24, 41, 14, 4, 18, 7, 2]
sample_500 = [5, 68, 48, 11, 25, 37, 3, 25, 5, 60, 106, 36, 6, 46, 16, 3]
sample_1000 = [10, 136, 96, 22, 50, 74, 6, 50, 10, 120, 212, 72, 12, 92, 32, 6]
rsample_200 = [1, 28, 16, 5, 9, 14, 1, 9, 1, 19, 47, 12, 4, 24, 8, 2]

# 20%:10%:70% data for training, validation and testing
VALIDATION_SPLIT = 0.85 
all_indices, train_indices, test_indices = sampling(VALIDATION_SPLIT, gt, sample_500)

# Getting training testing all datasets

train_data, test_data, all_data = Getting_HSIsets(train_indices, test_indices, all_indices,\
                                                  PATCH_LENGTH, INPUT_DIMENSION_CONV, whole_data, padded_data)

y_train = gt[train_indices] - 1
y_test = gt[test_indices] - 1
y_all = gt[all_indices] - 1
'''
train_spc = spectral_matrixs(train_data)

test_spc = spectral_matrixs(test_data)

all_spc = spectral_matrixs(all_data)
'''
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device('cuda', args.gpu)
#torch.cudnn.benchmark = True

# Parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8}

# Generators
'''
training_set = Hybrid_HSIDataset(range(len(train_indices)), train_data, y_train, train_spc)
training_generator = data.DataLoader(training_set, **params)

validation_set = Hybrid_HSIDataset(range(len(test_indices)), test_data, y_test, test_spc)
validation_generator = data.DataLoader(validation_set, **params)

all_set = Hybrid_HSIDataset(range(len(all_indices)), all_data, y_all, all_spc)
all_generator = data.DataLoader(all_set, **params)
'''

training_set = HSIDataset(range(len(train_indices)), train_data, y_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = HSIDataset(range(len(test_indices)), test_data, y_test)
validation_generator = data.DataLoader(validation_set, **params)

all_set = HSIDataset(range(len(all_indices)), all_data, y_all)
all_generator = data.DataLoader(all_set, **params)


trainloader = torch.utils.data.DataLoader(training_set, batch_size=50, shuffle=True, num_workers=0)

validationloader = torch.utils.data.DataLoader(validation_set, batch_size=50, shuffle=False, num_workers=0)

allloader = torch.utils.data.DataLoader(all_set, batch_size=50, shuffle=False, num_workers=0)


#allloader = torch.utils.data.DataLoader(all_set, batch_size=50, shuffle=False, num_workers=0)

net = HSI_Mixer_Net(num_classes = nb_classes, img_size = PATCH_LENGTH*2+1, patch_size = 3, depth = args.depth, in_chans = INPUT_DIMENSION_CONV, embed_dim=28)
net.to(device)
summary(net, input_shape=(200, 9, 9))

criterion = nn.CrossEntropyLoss()
#optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

best_pred = 0
# #SAVE_PATH3 = './saved_models/ssnet_best3_up_seed' + str(args.seed) + '.pth' 
SAVE_PATH3 = args.save + '/modules'+ str(args.dataset) + str(args.model) + '_seed' + str(args.seed) + '.pth' 
# #torch.save(net.state_dict(), SAVE_PATH)

for epoch in range(args.epochs):  # loop over the dataset multiple times
    
    running_loss = 0.0
    #iters = len(trainloader)
    net = net.train()
    for i, batch_data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:    # print every 2000 mini-batches
            logging.info('[%d, %5d] loss: %.4f' %
                  (epoch + 1, i + 1, running_loss / 4))
            running_loss = 0.0
    #schedular.step()
    
    correct = 0
    total = 0
    net = net.eval()
    counter = 0 
    with torch.no_grad():
        for batch_data in validationloader:
#             if counter <= 10:
#                 counter += 1
            images, labels = batch_data
            images, labels,  = images.to(device), labels.to(device)
            outputs = net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

    new_pred = correct / total
    logging.info('Accuracy of the network on the validation set: %.5f %%' % (
        100 * new_pred))
    
    if new_pred > best_pred:
        logging.info('new_pred %f', new_pred)
        logging.info('best_pred %f', best_pred)
        torch.save(net.state_dict(), SAVE_PATH3)
        best_pred=new_pred
        
logging.info('Finished Training')

trained_net = HSI_Mixer_Net(num_classes = 16, img_size = PATCH_LENGTH*2+1, patch_size = 3, depth = args.depth, in_chans = INPUT_DIMENSION_CONV, embed_dim=28)
trained_net.load_state_dict(torch.load(SAVE_PATH3))
trained_net.eval()
trained_net = trained_net.cuda()

label_val = []
pred_val = []

with torch.no_grad():
    for batch_data in validationloader:
        images, labels = batch_data
        #label_val = torch.stack([label_val.type_as(labels), labels])
        label_val.append(labels)
        
        images, labels = images.to(device), labels.to(device)
        outputs = trained_net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        #pred_val = torch.stack([pred_val.type_as(predicted), predicted])
        pred_val.append(predicted)
        
label_val_cpu = [x.cpu() for x in label_val]
pred_val_cpu = [x.cpu() for x in pred_val]

label_cat = np.concatenate(label_val_cpu)
pred_cat = np.concatenate(pred_val_cpu)

matrix = metrics.confusion_matrix(label_cat, pred_cat)

OA, AA_mean, Kappa, AA = cal_results(matrix)

logging.info('OA, AA_Mean, Kappa with validation: %f, %f, %f, ', OA, AA_mean, Kappa)
logging.info(str(("AA for each class: ", AA)))

label_all = []
pred_all = []

with torch.no_grad():
    for batch_data in allloader:
        images, labels = batch_data
        #label_val = torch.stack([label_val.type_as(labels), labels])
        label_all.append(labels)
        
        images, labels = images.to(device), labels.to(device)
        outputs = trained_net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        #pred_val = torch.stack([pred_val.type_as(predicted), predicted])
        pred_all.append(predicted)
        
label_all_cpu = [x.cpu() for x in label_all]
pred_all_cpu = [x.cpu() for x in pred_all]

label_cat_all = np.concatenate(label_all_cpu)
pred_cat_all = np.concatenate(pred_all_cpu)

matrix = metrics.confusion_matrix(label_cat_all, pred_cat_all)

OA, AA_mean, Kappa, AA = cal_results(matrix)

logging.info('OA, AA_Mean, Kappa with allpred: %f, %f, %f, ', OA, AA_mean, Kappa)
logging.info(str(("AA for each class: ", AA)))

# generate classification maps

all_pred = []

with torch.no_grad():
    for batch_data in allloader:
        images, _ = batch_data
        images, _ = images.to(device), labels.to(device)
        outputs = trained_net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        all_pred.append(predicted)

all_pred = torch.cat(all_pred)
all_pred = all_pred.cpu().numpy() + 1

y_pred = predVisIN(all_indices, all_pred, 'IN', 145, 145)

all_pred1 = []
ALL_SIZE = data_hsi.shape[1]
all_size_indices = np.arange(ALL_SIZE)

all_size_data = np.zeros((ALL_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
all_size_assign = indexToAssignment(range(ALL_SIZE), PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
for i in range(len(all_size_assign)):
    all_size_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_size_assign[i][0], all_size_assign[i][1])

all_size_set = all_HSI_Data_pred(range(len(all_size_indices)), all_size_data)
all_size_generator = data.DataLoader(all_size_set, **params)
all_sizeloader = torch.utils.data.DataLoader(all_size_set, batch_size=50, shuffle=False, num_workers=0)

with torch.no_grad():
    for batch_data in all_sizeloader:
        images = batch_data
        images = images.to(device)
        outputs = trained_net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        all_pred1.append(predicted)

all_pred1 = torch.cat(all_pred1)
all_pred1 = all_pred1.cpu().numpy() + 1

y_pred1 = predVisIN(all_size_indices, all_pred1, 'IN', 145, 145)



# #plt.plot(x, y)
plt.imshow(y_pred)
plt.axis('off')
fig_path = './saved_figures/' + 'IN' + '_seed' + str(args.seed) + '.png'
plt.savefig(fig_path, bbox_inches=0, dpi=300)
#plt.savefig(fig_path, bbox_inches='tight')

plt.imshow(y_pred1)
plt.axis('off')
fig_path = './saved_figures/' + 'IN1' + '_seed' + str(args.seed) + '.png'
plt.savefig(fig_path, bbox_inches=0, dpi=300)
