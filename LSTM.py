import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import data_utils
import scipy.io as sio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
num_resids = 6
batch_size = 128
decay_bound_epochs = [60, 100, 140]
decay_rates = [1, 0.1, 0.01, 0.001]
momentum = 0.9
dropout = 0
hidden_size = 100
num_features = 32
window_size= 20
save = './final_win%d_b%d_mse' % (window_size,batch_size)
print(save)

parser = argparse.ArgumentParser()
# they used network to choose between rnn and node. we are separate files
#parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')

# tol is both the relative and absolute tolerance
parser.add_argument('--tol', type=float, default=1e-3)

# adjoint is a parameter of ODE. It has to do with back prop
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])

# They used conv to down sample. We don't need down sampling
#parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])

# Number of training epochs
parser.add_argument('--nepochs', type=int, default=20)

# Whether or not we augment data. May not need this
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])

# Rest are obvious
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default=save)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


# Below is from Google Seq2Seq. Starting with a simplified version of this for our work
"""
class StackedCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.LSTMCell, residual=False):
        super(StackedCell, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.residual = residual
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            rnn = rnn_cell(input_size, hidden_size, bias=bias)
            self.layers.append(rnn)
            input_size = hidden_size

    def forward(self, inputs, hidden):
        def select_layer(h_state, i):  # To work on both LSTM / GRU, RNN
            if isinstance(h_state, tuple):
                return tuple([select_layer(s, i) for s in h_state])
            else:
                return h_state[i]

        next_hidden = []
        for i, layer in enumerate(self.layers):
            next_hidden_i = layer(inputs, select_layer(hidden, i))
            output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
                else next_hidden_i
            if i + 1 < self.num_layers:
                output = self.dropout(output)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden
"""

# Additional Reference:https://medium.com/coinmonks/character-to-character-rnn-with-pytorchs-lstmcell-cd923a6d0e72
class SingleLSTMResidual(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, seq_len, num_feats=32,output_dim=1, bias=True, dropout=0):
        super(SingleLSTMResidual, self).__init__()
        self.hidden_dim = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.input_size = input_size
        #self.layer = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=2,batch_first=True)
        self.layer1 = nn.LSTMCell(input_size, hidden_size, bias=bias)
        self.layer2 = nn.LSTMCell(hidden_size,hidden_size)
        self.linear = nn.Linear(hidden_size,1)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(num_feats)

    def forward(self, input, z, hc):
        # Input is expected to be T, N, D where N is batchsize. T is the window and D is number of features
        outputs = torch.empty((self.seq_len, self.batch_size, self.hidden_dim)).to(device)

        h1,c1 = hc
        h2,c2 = hc


        # loop backwards in time
        this_in = torch.empty((self.batch_size,self.input_size))
        for i in reversed(range(self.seq_len)):
            this_in = torch.cat((input[:, i, :], z[:, :, 0]), dim=1)
            #for j in range(1,3):
            #   this_in = torch.cat((this_in,z[:,:,j]),dim=1)

            h1, c1 = self.layer1(this_in, (h1,c1))
            h2, c2 = self.layer2(h1, (h2,c2))
            outputs[i] = h2

        last_out = outputs[-1]
        output = self.linear(last_out)

        return output

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.batch_size, self.hidden_dim).to(device))
        #return (torch.zeros(2,self.batch_size,self.hidden_dim).to(device),torch.zeros(2,self.batch_size,self.hidden_dim).to(device))
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

#TODO UPDATE
def accuracy(model, dataset_loader):
    run_error = 0
    for x, y, z in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        hc = LSTM_layer.init_hidden()
        predict = model(x,z,hc)

        run_error += (predict - y)**2

    run_error = run_error / len(dataset_loader)
    rmse = torch.sqrt(torch.mean(run_error))
    return rmse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


# important hyper parameters

# GPU
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # Set up saving stuff
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    # Create Residual network.

    #feature_layers = [ResBlock(64, 64) for _ in range(num_resids)]
    #fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(64, 10)]

    #model = nn.Sequential(*feature_layers, *fc_layers).to(device)

    criterion = nn.MSELoss().to(device)

    # Get data loaders
    # Parameters:
    # data_utils.z_1 (keep this, this just means it'll use the first dataset (thumb)
    # num_labels:1 (keep this, the first dataset has only 1 label and that it thumb
    # which_feats: list of which features you want to use. [0,1,2,...,7] means use features 0 to 7 only (MAV features)
    #              [0,1,2,...,32] means use features 0 to 32 (All features
    # window_size: the size of the sliding window. sliding window = 100 means that the model recieves the last 100 time
    #              points when it tries to predict the current timepoint's label
    # batch_size:  the usual meaning of batch size
    # Center:      True if you want to zero center the labels. False otherwise.
    which_feats = [0,1,2,3,4,5,6,7] # MAV FEATURES ONLY
    fingers = [data_utils.THUMB_INDEX]#,data_utils.INDEX_INDEX,data_utils.MIDDLE_INDEX]
    num_labels = 1
    center = False
    train_loader, test_loader, train_eval_loader = data_utils.get_data_loaders(
        data_utils.z_1,fingers,num_labels,which_feats,window_size,batch_size,shuffle=True,center=center)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    LSTM_layer = SingleLSTMResidual(input_size=8+num_labels*(window_size-1), seq_len=window_size,hidden_size=hidden_size, dropout=dropout, batch_size=batch_size)
    model = LSTM_layer.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))
    best_acc = 100000
    #hidden = LSTM_layer.init_hidden().to(device)
    #o_plots = []
    #y_plots = []
    for i in range(args.nepochs):
        for x,y,z in train_loader:
            optimizer.zero_grad()
            # Shape of x is (Batchsize, Windowlength, numfeats). Reshape if you need to.
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            hc = LSTM_layer.init_hidden()

            output = model(x, z, hc)
            # MAKE SURE YOUR MODEL ONLY RETURNS 1 LABEL. CHECK THE FORWARD FUNCTION, ITS LIKELY THAT IT RETURNS A SHAPE
            # OF (Windowlength, 1). IF THIS IS THE CASE, TAKE THE LAST LABEL (i.e. output = output[-1].
            #o_plots.extend(output.cpu().detach().numpy())
            #y_plots.extend(y.cpu().detach().numpy())
            loss = criterion(output,y)
            #loss = torch.mean(torch.abs(output - y))
            #loss = torch.mean(torch.abs(output-y))
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            train_acc = accuracy(model, train_eval_loader)
            val_acc = accuracy(model, test_loader)
            print(i,': ',train_acc,val_acc)
            if val_acc < best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                best_acc = val_acc
                print("Saved")
