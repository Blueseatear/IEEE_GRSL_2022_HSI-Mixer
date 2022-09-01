# -*- coding: utf-8 -*-
"""
Created on Mon May 23 21:37:24 2022

@author: Blues
"""

from torch.utils import data
import numpy as np

def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:,range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, :, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def old_sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        print("%d th catetory: %d" % (len(indices), nb_val))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return whole_indices, train_indices, test_indices

def sampling(proptionVal, groundTruth, sample_num=None):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    val = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        if sample_num:
            train[i] = indices[-sample_num[i]:]
            #train2[i] = indices[sample_num[i]:sample_num[i]+rsample_num[i]]
            val[i] =  indices[-(sample_num[i]+sample_num[i]):-sample_num[i]]
            test[i] = indices[:-(sample_num[i])]
        else:
            nb_val = int(proptionVal * len(indices))
            train[i] = indices[:-nb_val]
            test[i] = indices[-nb_val:]
    whole_indices = []
    train_indices = []
    test_indices = []
    #val_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
        #val_indices += val[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return whole_indices, train_indices, test_indices

def rsampling(groundTruth, sample_num, rsample_num):              #divide dataset into train and test datasets
    labels_loc = {}
    labeled = {}
    train2 = {}
    val = {}
    test = {}
    m = np.max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        labeled[i] = indices[:sample_num[i]]
        train2[i] = indices[sample_num[i]:sample_num[i]+rsample_num[i]]
        val[i] =  indices[-(sample_num[i]+rsample_num[i]):]
        test[i] = indices[sample_num[i]+rsample_num[i]:-(sample_num[i]+rsample_num[i])]
    whole_indices = []
    labeled_indices = []
    train2_indices = []
    val_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        labeled_indices += labeled[i]
        train2_indices += train2[i]
        val_indices += val[i]
        test_indices += test[i]
        np.random.shuffle(labeled_indices)
        np.random.shuffle(train2_indices)        
        np.random.shuffle(val_indices)        
        np.random.shuffle(test_indices)
    return whole_indices, labeled_indices, train2_indices, val_indices, test_indices

def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix

def Getting_HSIsets(train_indices, test_indices, all_indices, PATCH_LENGTH, INPUT_DIMENSION_CONV, whole_data, padded_data):
    TOTAL_SIZE = len(all_indices)
    print("TOTAL SAMPLE:", TOTAL_SIZE)
    
    TRAIN_SIZE = len(train_indices) #300
    print("TRAIN SAMPLE:", TRAIN_SIZE)
    
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print("TEST SAMPLE:", TEST_SIZE)
    
    train_data = np.zeros((TRAIN_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    test_data = np.zeros((TEST_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    all_data = np.zeros((TOTAL_SIZE, INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    
    train_assign = indexToAssignment(train_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, train_assign[i][0], train_assign[i][1])
    
    test_assign = indexToAssignment(test_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, test_assign[i][0], test_assign[i][1])
        
    all_assign = indexToAssignment(all_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for i in range(len(all_assign)):
        all_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_assign[i][0], all_assign[i][1])
    
    return train_data, test_data, all_data

class HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples, labels):
        
        self.list_IDs = list_IDs
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        y = self.labels[ID]

        return X, y

class all_HSI_Data_pred(data.Dataset):
    def __init__(self, list_IDs, samples):
        
        self.list_IDs = list_IDs
        self.samples = samples

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]

        return X
        
class Hybrid_HSIDataset(data.Dataset):
    def __init__(self, list_IDs, samples, labels, spc_matrixs):
        
        self.list_IDs = list_IDs
        self.samples = samples
        self.labels = labels
        self.spc_matrixs = spc_matrixs
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        y = self.labels[ID]
        z = self.spc_matrixs[ID]
        return X, y, z