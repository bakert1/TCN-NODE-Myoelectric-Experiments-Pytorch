import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as tdata
from torch.utils.data import DataLoader
import sys
sys.path.append("../../")
from TCN.EMG.model import TCN
from TCN.EMG.utils import data_generator
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline


parser = argparse.ArgumentParser(description='Sequence Modeling - EMG Signal')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size (default: 20)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: FALSE)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit (default: 5)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--seq_len', type=int, default=32,
                    help='sequence length (default: 32)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='report interval (default: 1000')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=50,
                    help='number of hidden units per layer (default: 50)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()


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
    def __init__(self, data, num_labels, which_feats, window_size, center=False, delta=False):
        super(FingerDataset, self).__init__()
        N,M = data.shape
        labels = data[:,0:num_labels] # The labels are expected first
        features = data[:,num_labels:] # The features follow
        features = features[:,which_feats] # only grab the features that you want (i.e. features [0,1,...,7] are MAV

        # zero center labels
        if center:
            labels = labels - np.mean(labels)

        start = 0
        if delta:
            for i in range(1,len(labels)):
                labels[i] = labels[i] - labels[i-1]
            start = 1

        self.features = features[start:]
        self.window = window_size
        self.labels = labels[start:]


    def __len__(self):
        return len(self.features) - self.window + 1

    def __getitem__(self,index):
        #return (self.features[index:index+self.window], self.labels[index+self.window-1], self.labels[index+self.window-2])
        return (self.features[index:index+self.window], self.labels[index+self.window-1], self.labels[index:index+self.window-1])
    
    
def preprocess_data(z , fingers, skip_first = True):
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


def get_data_loaders(z, fingers, num_labels, which_feats, window_size, batch_size, train_split=0.8, validation_split=0.2, center =True, shuffle=True):
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
    data = preprocess_data(z, fingers, True)
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
            DataLoader(FingerDataset(dataset,num_labels,which_feats,window_size,center), batch_size=batch_size,drop_last=True,shuffle=shuffle)
        )

    return loaders

def RMSE_error(model, dataset_loader): # should be named error
    RMSE = 0
    
    with torch.no_grad():
        for x, y, z in dataset_loader:
            x = torch.tensor(x).cuda()
            y = torch.tensor(y).cuda()  
            z = torch.tensor(z).cuda()

            new_input = torch.empty((batch_size, window_size, window_size+8-1)).cuda()
            for i in range(window_size):
                new_input[:, i, :] = torch.cat((x[:, i, :], z[:, :, 0]), dim=1)

            output = model(new_input)

            error = (output - y).squeeze()
            RMSE += ((error.pow(2)).mean()).sqrt()

    return RMSE.item() / len(dataset_loader)


def evaluate():
    ground_truths = []
    predicted = []
    total_loss = 0
    with torch.no_grad():
        for x, y, z in test_loader_unshuffled:
            x = torch.tensor(x).cuda()
            y = torch.tensor(y).cuda()  
            z = torch.tensor(z).cuda()

            new_input = torch.empty((batch_size, window_size, window_size+8-1)).cuda()
            for i in range(window_size):
                new_input[:, i, :] = torch.cat((x[:, i, :], z[:, :, 0]), dim=1)

            output = model(new_input)
            loss = abs(output - y).mean()
            total_loss += loss.item()

            predicted.extend(output)
            ground_truths.extend(y)

    plt.plot(predicted, color = 'red', label = 'Temporal CNN Prediction')
    plt.plot(ground_truths, color = 'blue', label = 'Ground Truth')
    plt.show()
    
    predicted = torch.tensor(predicted).cuda()
    ground_truths = torch.tensor(ground_truths).cuda()
    score_matric = 1 - (predicted - ground_truths).pow(2).sum() / (ground_truths - ground_truths.mean()).pow(2).sum()
    print("Goodness of fit = ", score_matric.item())
    print("Loss = ", total_loss / len(test_loader_unshuffled))
    
    
all_data = sio.loadmat('z.mat', struct_as_record=False, squeeze_me=True)
z = all_data['z']
THUMB_INDEX = 0
INDEX_INDEX = 1
MIDDLE_INDEX = 2
RING_INDEX = 3
PINKY_INDEX = 4

num_labels = 1
which_feats = [0, 1, 2, 3, 4, 5, 6, 7] # MAV Featrues ONLY
window_size = 20
batch_size = 20
center = False

train_loader, test_loader, train_eval_loader = get_data_loaders(
    z, [0], num_labels, which_feats, window_size, batch_size, center=center, shuffle=True)
train_loader_unshuffled, test_loader_unshuffled, train_eval_loader_unshuffled = get_data_loaders(
    z, [0], num_labels, which_feats, window_size, batch_size, center=center, shuffle=False)
train_loader_91, test_loader_91, train_eval_loader_91 = get_data_loaders(
    z, [0], num_labels, which_feats, window_size, batch_size, train_split=0.9, validation_split=0.1, center=center, shuffle=False)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

input_channels = 1
n_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs

print(args)
all_data = sio.loadmat('z.mat', struct_as_record=False, squeeze_me=True)
# all_data = sio.loadmat('/content/gdrive/My Drive/data/z.mat', struct_as_record=False, squeeze_me=True)
struct = all_data['z']
THUMB_INDEX = 0
def preprocess_data(z , finger=THUMB_INDEX):
    '''
    :param z: data of with three dimensions where:
              Dimension 1 refers to which trial
              Dimension 2 refers to which time point in a trial
              Dimension 3 refers to which data point of a time point (the first data point is the finger flexion and
                          the following 32 data points are the 32 features for that time point)

    :return: data of shape (T, D) where:
              T = sum of number of time points in each trial (each trial has a different number of time points)
              D = 33 = number of data points of time point (the first data point is the finger flexion and
                          the following 32 data points are the 32 features for that time point)
    '''
    new_z = None
    for trial in z:
        angles = trial.FingerAnglesTIMRL[:,finger]
        feats = trial.EMGFeats
        data = np.c_[angles,feats]
        if new_z is None:
            new_z = data
        else:
            new_z = np.r_[new_z, data]

    return new_z

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(input_channels, n_classes, channel_sizes, window_size, 1, kernel_size=kernel_size, dropout=dropout)

if args.cuda:
    model.cuda()
#    X_train = torch.tensor(X_train).cuda()
#    Y_train = torch.tensor(Y_train).cuda()
#    X_test = torch.tensor(X_test).cuda()
#    Y_test = torch.tensor(Y_test).cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(epoch):
    total_output = []
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    
    for i, (x, y, z) in enumerate(train_loader):
        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()  
        z = torch.tensor(z).cuda()
        
        new_input = torch.empty((batch_size, window_size, window_size+8-1)).cuda()
        for j in range(window_size):
            new_input[:, j, :] = torch.cat((x[:, j, :], z[:, :, 0]), dim=1)
        
        optimizer.zero_grad()
        output = model(new_input)
        total_output.extend(output)
        
        #loss = F.mse_loss(output, y)
        loss = abs(output - y).mean()
        #loss += (reg*abs(output - z[:, -1, :])).mean()
        #if batch_idx % log_interval == 0:
        #    print((reg*abs(output - z)).mean())
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            processed = min(i+batch_size, len(train_loader) * batch_size)
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, len(train_loader), 100.*processed/len(train_loader), lr, np.sqrt(cur_loss)))
            total_loss = 0


for ep in range(1, epochs+1):
    train(ep)
    evaluate()

ground_truths = []
predicted = []
total_loss = 0

with torch.no_grad():
    for x, y, z in test_loader_unshuffled:
        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()  
        z = torch.tensor(z).cuda()

        new_input = torch.empty((batch_size, window_size, window_size+8-1)).cuda()
        for j in range(window_size):
            new_input[:, j, :] = torch.cat((x[:, j, :], z[:, :, 0]), dim=1) 

        output = model(new_input)
        loss = abs(output - y).mean()
        #loss += (reg*abs(output - z[:, -1, :])).mean()
        total_loss += loss.item()

        predicted.extend(output)
        ground_truths.extend(y)

plt.plot(predicted, color = 'red', label = 'Temporal CNN Prediction')
plt.plot(ground_truths, color = 'blue', label = 'Ground Truth')
plt.xlabel('Time(ms)') 
plt.ylabel('Percent Finger Flexion')
plt.title('TCNN Prediction')
plt.ylim(0.2, 0.8)
plt.legend()
plt.show()

predicted = torch.tensor(predicted).cuda()
ground_truths = torch.tensor(ground_truths).cuda()
score_matric = 1 - (predicted - ground_truths).pow(2).sum() / (ground_truths - ground_truths.mean()).pow(2).sum()
print("Goodness of fit = ", score_matric.item())
print("Train RMSE = ", RMSE_error(model, train_loader))
print("Test RMSE = ", RMSE_error(model, test_loader_unshuffled))
print("Loss = ", total_loss / len(test_loader_unshuffled))


all_data1 = sio.loadmat('Z_run-010_thumb_index_middle.mat',
                           struct_as_record=False, squeeze_me=True)
z1 = all_data1['z']
finger_strings = ["Thumb","Index","Middle","Ring","Pinky"]
fingers = [THUMB_INDEX, INDEX_INDEX, MIDDLE_INDEX]
num_labels_3_fingers = 3
window_size1 = 20
input_channels = window_size1

train_loader3, test_loader3, train_eval_loader3 = get_data_loaders(
    z1, fingers, num_labels_3_fingers, which_feats, window_size1, batch_size, center=center, shuffle=True)
train_loader3_unshuffled, test_loader3_unshuffled, train_eval_loader3_unshuffled = get_data_loaders(
    z1, fingers, num_labels_3_fingers, which_feats, window_size1, batch_size, center=center, shuffle=False)

def evaluate1():
    ground_truths = []
    predicted = []
    #total_loss = 0
    
    with torch.no_grad():
        for x, y, z in test_loader3_unshuffled:
            x = torch.tensor(x).cuda()
            y = np.array(y.numpy())
            z = torch.tensor(z).cuda()

            new_input = torch.empty((batch_size, window_size1, 3*(window_size1-1)+8)).cuda()
            tmp_z = torch.flatten(z, start_dim=1)
            for j in range(window_size1):
                new_input[:, j, :] = torch.cat((x[:, j, :], tmp_z), dim=1)

            output = model1(new_input).cpu().detach().numpy()
            #loss = abs(output - y).mean()
            #total_loss += loss.item()

            predicted.extend(output)
            ground_truths.extend(y)
    
    predicted = np.asarray(predicted)
    ground_truths = np.asarray(ground_truths)
    fig = plt.figure(figsize=(20,3))
    for i in range(len(fingers)):
        plt.subplot(1, 3, i+1)
        plt.plot(predicted[:, i], color = 'red', label = 'Temporal CNN Prediction')
        plt.plot(ground_truths[:, i], color = 'blue', label = 'Ground Truth')
        plt.xlabel('Time(ms)') 
        plt.ylabel('Percent Finger Flexion of %s' % finger_strings[i])
        plt.title('TCNN Prediction')
    plt.show()
    
    predicted = torch.tensor(predicted).cuda()
    ground_truths = torch.tensor(ground_truths).cuda()
    score_matric = 1 - (predicted - ground_truths).pow(2).sum() / (ground_truths - ground_truths.mean()).pow(2).sum()
    print("Goodness of fit = ", score_matric.item())
    print("Loss = ", total_loss / len(test_loader_unshuffled))
    
    
model1 = TCN(input_channels, 3, channel_sizes, window_size, 3, kernel_size=kernel_size, dropout=dropout)

if use_cuda:
    model1.cuda()

optimizer = optim.Adam(model1.parameters(), lr=lr)

def train(epoch):
    total_output = []
    global lr
    model1.train()
    batch_idx = 1
    total_loss = 0

    for i, (x, y, z) in enumerate(train_loader3):
        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()  
        z = torch.tensor(z).cuda()
        
        new_input = torch.empty((batch_size, window_size1, 3*(window_size1-1)+8)).cuda()
        tmp_z = torch.flatten(z, start_dim=1)
        for j in range(window_size1):
            new_input[:, j, :] = torch.cat((x[:, j, :], tmp_z), dim=1)
        
        optimizer.zero_grad()
        output = model1(new_input)
        total_output.extend(output)
        
        #loss = F.mse_loss(output, y)
        loss = abs(output - y).mean()
        #loss += (reg*abs(output - z[:, -1, :])).mean()
        #if batch_idx % log_interval == 0:
        #    print((reg*abs(output - z)).mean())
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model1.parameters(), clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            processed = min(i+batch_size, len(train_loader3) * batch_size)
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, len(train_loader3), 100.*processed/len(train_loader3), lr, np.sqrt(cur_loss)))
            total_loss = 0

for ep in range(1, epochs+1):
    train(ep)
    evaluate1()
    
ground_truths = []
predicted = []
total_loss = 0

with torch.no_grad():
    for x, y, z in test_loader3_unshuffled:
        x = torch.tensor(x).cuda()
        y = np.array(y.numpy())
        z = torch.tensor(z).cuda()

        new_input = torch.empty((batch_size, window_size, 3*(window_size-1)+8)).cuda()
        tmp_z = torch.flatten(z, start_dim=1)
        for j in range(window_size):
            new_input[:, j, :] = torch.cat((x[:, j, :], tmp_z), dim=1)

        output = model1(new_input).cpu().detach().numpy()
        loss = abs(output - y).mean()
        total_loss += loss.item()

        predicted.extend(output)
        ground_truths.extend(y)

predicted = np.asarray(predicted)
ground_truths = np.asarray(ground_truths)
fig = plt.figure(figsize=(20,4))
for i in range(len(fingers)):
    plt.subplot(1, 3, i+1)
    plt.plot(predicted[:, i], color = 'red', label = 'Temporal CNN Prediction')
    plt.plot(ground_truths[:, i], color = 'blue', label = 'Ground Truth')
    plt.xlabel('Time(ms)') 
    plt.ylabel('Percent Finger Flexion of %s' % finger_strings[i])
    plt.title('TCNN Prediction')
    plt.legend()
plt.show()
fig.savefig('TCN_results_multi.png', dpi=fig.dpi)

predicted = torch.tensor(predicted).cuda()
ground_truths = torch.tensor(ground_truths).cuda()
score_matric = 1 - (predicted - ground_truths).pow(2).sum() / (ground_truths - ground_truths.mean()).pow(2).sum()
print("Goodness of fit = ", score_matric.item())
print("Loss = ", total_loss / len(test_loader_unshuffled))

def RMSE_error1(model, dataset_loader): # should be named error
    RMSE = 0
    
    with torch.no_grad():
        for x, y, z in dataset_loader:
            x = torch.tensor(x).cuda()
            y = torch.tensor(y).cuda()  
            z = torch.tensor(z).cuda()

            new_input = torch.empty((batch_size, window_size, 3*(window_size-1)+8)).cuda()
            tmp_z = torch.flatten(z, start_dim=1)
            for j in range(window_size):
                new_input[:, j, :] = torch.cat((x[:, j, :], tmp_z), dim=1)

            output = model(new_input)

            error = (output - y).squeeze()
            RMSE += ((error.pow(2)).mean()).sqrt()

    return RMSE.item() / len(dataset_loader)


predicted = torch.tensor(predicted).cuda()
ground_truths = torch.tensor(ground_truths).cuda()
score_matric = 1 - (predicted - ground_truths).pow(2).sum() / (ground_truths - ground_truths.mean()).pow(2).sum()
print("Goodness of fit = ", score_matric.item())
print("Train RMSE = ", RMSE_error1(model1, train_loader3))
print("Test RMSE = ", RMSE_error1(model1, test_loader3))
print("Loss = ", total_loss / len(test_loader3_unshuffled))

