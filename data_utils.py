import scipy.io as sio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from enum import Enum
import torch.utils.data as tdata
from torch.utils.data import DataLoader
import torch
import LSTM
import torch.nn as nn

mpl.rcParams.update({'font.size': 10})

# Constants
DATA_DIR = "data"
DATA_SET1 = "z_1.mat"
DATA_SET2 = "Z_run-010_thumb_index_middle.mat"
DATA_SET3 = "Z_run-017_index_middle.mat"
DATA_SET4 = "Z_run-019_index_middle.mat"

# The data stores data about each finger using the following indexing for (thumb, index, middle, ring, pinky)
THUMB_INDEX = 0
INDEX_INDEX = 1
MIDDLE_INDEX = 2
RING_INDEX = 3
PINKY_INDEX = 4

# Datas:
z_1 = sio.loadmat(os.path.join(DATA_DIR, DATA_SET1), struct_as_record=False, squeeze_me=True)['z']
# The data stores features from each of the 8 channels (channel refers to EMG data from specified muscle)
# Python recommends using classes to create enumerations
# Ch1: EPB (thumb extensor)
# Ch2: FPL  (thumb flexor)
# CH3: EIP (thumb extensor)
# CH4: FDS index (index flexor)
# CH5: FDP index (index flexor)
# CH6: FDS middle (middle flexor)
# CH7: FDP middle (middle flexor)
# CH8: EDC (all 4 fingers extensor)
class Feats(Enum):
    pass



class FingerDataset(tdata.Dataset):
    '''
    Torch Dataset class for handling the finger data. ONLY TESTED FOR OUR FIRST DATASET

    Batches data with a sliding window. The size of the sliding window determines how much information the networks
    can work with for each prediction. E.g. a window size of 50 means the network works with the last 50 ms of features
    to predict the finger angle at the current moment of time. The Northwestern paper discusses this more in depth.

    Expected format of data:
    First dimension: Corresponds to time points
    Second dimension: Corresponds to labels and features. The label(s) are first then features follow (e.g. for first
    data set, we had to label so data[:,0] is the labels and data[:,1:] are the features

    N: Number of time points
    D: Number of features
    T: Window size
    L: Number of labels

    self.features.shape: (N,D)
    self.labels.shape: (N,L)
    self.window: T

    len(self): N - T + 1
    self[i]: returns tuple, first dim is features, second dim is labels
    self[i][0].shape = self[i][1].shape = (T,D)

    :param data: the state
    :param num_labels: the number of labels in the data.
    :param which_features: python list of which features you want
    :param window_size: size of the sliding window
    :param center: whether or not to zero center labels
    :param delta: whether or not to transform position labels to velocity labels
    '''
    def __init__(self, data, num_labels, which_feats, window_size, center=False):
        super(FingerDataset, self).__init__()
        N,M = data.shape
        labels = data[:,0:num_labels] # The labels are expected first

        features = data[:,num_labels:] # The features follow
        self.features = features[:,which_feats] # only grab the features that you want (i.e. features [0,1,...,8] are MAV


        # zero center labels
        if center:
            labels = labels - np.mean(labels)

        self.labels = labels
        self.window = window_size


    def __len__(self):
        return len(self.features) - self.window + 1

    def __getitem__(self,index):
        return (self.features[index:index+self.window], self.labels[index+self.window-1], self.labels[index:index+self.window-1])

def get_data_loaders(z, fingers,num_labels, which_feats, window_size, batch_size, train_split=0.8, validation_split=0.2, center =True):
    '''
    train_split: How much data is used for training versus testing
    validation_split: How much training data is used for validation
    center: Should the labels be centered around 0?

    Example: train_split = 0.9, validation_split=0.1, N data points:
    Training gets 90% of the data, but 10% of that is used for validation
    Testing gets the other 10% of data

    Train data: 81% of data
    Valid data: 9% of data
    Test data: 10% of data
    '''
    data = preprocess_data(z,fingers,True)
    T, _ = data.shape

    valid_end = int(T*train_split*validation_split)
    train_end = valid_end + int(T*train_split)

    valid_data = data[0:valid_end]
    train_data = data[valid_end:train_end]
    test_data = data[train_end:]

    #mu = np.mean(train_data[:,1:],axis=0)
    #sig = np.std(train_data[:,1:],axis=0)

    #train_data[:,1:] = (train_data[:,1:] - mu) / sig
    #valid_data[:,1:] = (valid_data[:,1:] - mu) / sig
    #test_data[:,1:] = (test_data[:,1:] - mu) / sig
    datas = [train_data, test_data, valid_data]
    loaders = []
    for dataset in datas:
        loaders.append(
            DataLoader(FingerDataset(dataset,num_labels,which_feats,window_size,center), batch_size=batch_size,drop_last=True,shuffle=False)
        )

    return loaders


def plot_feature_from_trials(z, first_trial, num_trials, finger=THUMB_INDEX, feat=0):
    ''' Plot the finger angles of a given finger. This is the topmost plot Scott showed us on Matlab.
    Each trial will be a different color
    :param z: the data struct given by the lab. z[i] returns the i-th trial
    :param first_trial: first trial to be plotted
    :param num_trials: number of consecutive trials to plot
    :param finger: which finger's angle should be plotted. Default is to plot the thumb's finger angle
    :param feat: feature to be plotted. feature 0 means FingerAngles. 1-33 corresponds to EMG Feats
    :return: None

    Tested by Tim: 3/17/2019
    '''

    x = 0
    for i in range(num_trials):
        if feat == 0:
            trial = z[first_trial + i].FingerAnglesTIMRL[:,finger]
        elif feat > 0 and feat < 33:
            trial = z[first_trial + i].EMGFeats[:,feat-1]
        else:
            raise "Feature must be between 0 and 33, but got %d in plot_feats_from_trials" % feat
        dx = len(trial)
        plt.plot(np.arange(x, x + dx), trial, color = 'r')
        x = x + dx

    plt.xlabel("Time (ms)")
    if feat == 0:
        plt.ylabel("Percent Flexion")
    else:
        plt.ylabel("Feat Num: %d" % feat)
    plt.tight_layout()
    plt.show()

def plot_channel_from_trials(z, first_trial, end_trial, channel=0):
    ''' Plots the channels 4 features
    Each trial will be a different color
    :param z: the data struct given by the lab. z[i] returns the i-th trial
    :param first_trial: first trial to be plotted
    :param end_trial: the last trial to be plotted
    :param channel: the channel whose features are to be plotted
    :return: None

    Tested by Tim: 3/23/2019
    '''
    num_trials = end_trial - first_trial + 1
    x = 0
    fig,axs = plt.subplots(nrows=2,ncols=2)
    i1 = channel * 4
    for i in range(num_trials):
        feats = [z[first_trial + i].EMGFeats[:,i1 + j] for j in range(4)]
        dx = len(feats[0]) # length of trial
        for j, feat in enumerate(feats):
            row = j // 2
            col = j % 2
            axs[row,col].plot(np.arange(x, x + dx), feat, color = 'r')
        x = x + dx

    plt.xlabel("Time (ms)")
    #plt.ylabel("Channel Num: %d" % channel)
    plt.tight_layout()
    plt.show()

def plot_all_feats_from_trials(z, first_trial, end_trial):
    ''' Plots the channels 4 features
    Each trial will be a different color
    :param z: the data struct given by the lab. z[i] returns the i-th trial
    :param first_trial: first trial to be plotted
    :param end_trial: the last trial to be plotted
    :param channel: the channel whose features are to be plotted
    :return: None

    Tested by Tim: 3/23/2019
    '''
    num_trials = end_trial - first_trial + 1
    x = 0
    rows = 4
    cols = 8
    fig,axs = plt.subplots(nrows=rows, ncols=cols)
    for i in range(num_trials):
        feats = [z[first_trial + i].EMGFeats[:,j] for j in range(rows*cols)]

        dx = len(feats[0]) # length of trial
        for j, feat in enumerate(feats):
            row = j // cols
            col = j % cols
            axs[row,col].plot(np.arange(x, x + dx), feat, color = 'r')
        x = x + dx

    #plt.ylabel("Channel Num: %d" % channel)
    plt.show()

def preprocess_data(z , fingers, skip_first=True):
    ''' Pre-processes the data Phil gave us. Excludes first trial which has weird spike
    :param z: data of with three dimensions where:
              Dimension 1 refers to which trial
              Dimension 2 refers to which time point in a trial
              Dimension 3 refers to which data point of a time point (the first data point is the finger flexion and
                          the following 32 data points are the 32 features for that time point)

    :return: data of shape (T, D) where:
              T = sum of number of time points in each trial (each trial has a different number of time points)
              D = 33 = number of data points of time point (the first data point is the finger flexion and
                          the following 32 data points are the 32 features for that time point)

    Tested by Tim on 3/23/19
    '''
    new_z = None
    if skip_first:
        start = 1
    else:
        start = 0


    for trial in z[start:]:
        angles = trial.FingerAnglesTIMRL[:,fingers]
        feats = trial.EMGFeats
        data = np.c_[angles,feats]
        if new_z is None:
            new_z = data
        else:
            new_z = np.r_[new_z, data]

    return new_z

if __name__ == "__main__":
    z_2 = sio.loadmat(os.path.join(DATA_DIR,DATA_SET1), struct_as_record=False, squeeze_me=True)['z']
    fingers = [THUMB_INDEX,INDEX_INDEX,MIDDLE_INDEX]
    num_labels = 3
    which_feats = [0,1,2,3,4,5,6,7]
    window_size = 10
    batch_size = 32
    criterion = nn.MSELoss()

    # Uncomment below to visualize data
    #plt.plot(data[:,0])
    #plt.plot(data[:,1])
    #plt.plot(data[:,2])
    #plt.show()
    train_loader,_,_ = get_data_loaders(z_2,fingers,num_labels,which_feats,window_size,batch_size)
    for x,y,z in train_loader:
        fake_predictions = torch.rand(y.shape) # your model would go here. Model needs to output "num_labels" amount of predictions

        loss = criterion(fake_predictions,y)
        
        # This isn't the complete train loop, but this was just here to test
