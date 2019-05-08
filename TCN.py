import torch.nn as nn
from torch.nn.utils import weight_norm
from torch import nn
import data_utils
import torch
import argparse
import torch.optim as optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

parser = argparse.ArgumentParser(description='Sequence Modeling - EMG Signal')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size (default: 20)')
parser.add_argument('--gpu', type=int, default=0,
                    help='use CUDA (default: 0)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--nepochs', type=int, default=5,
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
parser.add_argument('--window_size', type=int, default=20,
                    help='window size (default: 20)')

args = parser.parse_args()

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, num_features, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear1 = nn.Linear(layer_size * num_channels[-1],
        #                          layer_size * num_channels[-1])
        self.linear1 = nn.Linear(num_features * num_channels[-1], output_size)

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        batch_size = x.size()[0]
        y1 = self.tcn(x)
        y1 = y1.reshape(batch_size, -1)
        # y2 = self.linear1(y1)
        output = self.linear1(y1)
        return output

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

def RMSE_error(model, dataset_loader):
    error = 0
    with torch.no_grad():
        for x, y, z in dataset_loader:
            x = x.to(device)
            y = np.array(y.numpy())
            z = z.to(device)

            new_input = torch.empty((batch_size, window_size, dimension)).to(device)
            z = torch.flatten(z, start_dim=1)
            for j in range(window_size):
                new_input[:, j, :] = torch.cat((x[:, j, :], z), dim=1)

            predicted = model(new_input).cpu().detach().numpy()

            error += (predicted - y)**2

    run_error = error / len(dataset_loader)

    RMSE = np.sqrt(np.mean(run_error))

    return RMSE

if __name__ == '__main__':

    args.save = './TestingTCN'

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # all_data = sio.loadmat('/Users/ScottEnsel/Desktop/Deep Learning/Project/NEW files/Z_run-010_thumb_index_middle.mat',
    #                        struct_as_record=False,
    #                        squeeze_me=True)
    all_data = sio.loadmat('/Users/ScottEnsel/Desktop/Deep Learning/Project/NEW files/Z_run-012_thumb.mat',
                           struct_as_record=False,
                           squeeze_me=True)
    EMG_data = all_data['z']

    THUMB_INDEX = 0
    INDEX_INDEX = 1
    MIDDLE_INDEX = 2
    RING_INDEX = 3
    PINKY_INDEX = 4
    ALL_FINGERS = [THUMB_INDEX, INDEX_INDEX, MIDDLE_INDEX, RING_INDEX, PINKY_INDEX]
    finger_strings = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    args.nepochs = 5
    args.window_size = 10
    args.batch_size = 32
    num_labels = 4

    which_feats = [0, 1, 2, 3, 4, 5, 6, 7] # MAV Features ONLY
    batch_size = args.batch_size
    window_size = args.window_size
    fingers = ALL_FINGERS[:num_labels]

    train_loader, test_loader, valid_loader = data_utils.get_data_loaders(EMG_data, fingers, num_labels,
                                                                          which_feats, args.window_size, args.batch_size,
                                                                          train_split=0.8, validation_split=0.2,
                                                                          center=False, shuffle = True)
    _, test_loader_unshuffled, _ = data_utils.get_data_loaders(EMG_data, fingers, num_labels,
                                                                          which_feats, args.window_size, args.batch_size,
                                                                          train_split=0.8, validation_split=0.2,
                                                                          center=False, shuffle=False)
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    dropout_ratio = 0.2
    clip = -1 #so no clipping
    seq_length = args.seq_len

    input_channels = args.window_size

    dimension = len(which_feats) + ((args.window_size - 1)*len(fingers))

    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    model = TCN(input_channels, num_labels, channel_sizes, dimension, kernel_size=kernel_size, dropout=dropout)

    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    log_interval = batches_per_epoch//4
    epoch_loss = 0
    num_processed = 0
    reset = 0
    best_err = float("inf")

    logger.info("training started")
    for itr in range(args.nepochs * batches_per_epoch):

        optimizer.zero_grad()
        x, y, z = data_gen.__next__()

        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        new_input = torch.empty((batch_size, window_size, dimension)).to(device)
        z = torch.flatten(z, start_dim=1)
        for j in range(window_size):
            new_input[:, j, :] = torch.cat((x[:, j, :], z), dim=1)

        predictions = model(new_input)

        loss = torch.mean(torch.abs(predictions - y))

        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

        num_processed += 1
        if itr % log_interval == (log_interval - 1):
            cur_loss = epoch_loss / num_processed
            logger.info(
                'Train Epoch: {:2d} [{:6d}/{:6d} ({:.2f}%)] |  Absolute Error Loss: {:.4f}'.format(
                (itr // batches_per_epoch)+1, num_processed, batches_per_epoch, 100*(num_processed/batches_per_epoch), cur_loss)
            )

        reset += 1
        if reset == batches_per_epoch: #reset counters
            reset = 0
            epoch_loss = 0
            num_processed = 0

        if itr % batches_per_epoch == (batches_per_epoch - 1):
            with torch.no_grad():
                train_err = RMSE_error(model, train_loader)
                val_err = RMSE_error(model, valid_loader)

                if val_err < best_err:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_err = val_err

                logger.info(
                    "Epoch {:02d} | Train RMSE {:.4f} | Validation RMSE {:.4f}".format(
                        (itr // batches_per_epoch)+1, train_err, val_err
                    )
                )

                # this is to visualize how training is going after each epoch
                ground_truths = []
                predicted = []
                for x, y, z in test_loader_unshuffled:
                    x = x.to(device)
                    y = np.array(y.numpy())
                    z = z.to(device)

                    new_input = torch.empty((batch_size, window_size, dimension)).to(device)
                    z = torch.flatten(z, start_dim=1)
                    for j in range(window_size):
                        new_input[:, j, :] = torch.cat((x[:, j, :], z), dim=1)

                    output = model(new_input).cpu().detach().numpy()

                    predicted.extend(output)
                    ground_truths.extend(y)

                predicted = np.asarray(predicted)
                ground_truths = np.asarray(ground_truths)
                for i in range(len(fingers)):
                    plt.subplot(1, len(fingers), i + 1)
                    plt.plot(predicted[:, i], color='red', label='TCN Prediction')
                    plt.plot(ground_truths[:, i], color='blue', label='Ground Truth')
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Percent Finger Flexion of %s' % finger_strings[i])
                    plt.title('TCN vs Ground Truth Epoch %.2d' % ((itr // batches_per_epoch)+1))
                    plt.legend()
                plt.show()

    logger.info("training finished")

    #now to plot final image
    ground_truths = []
    predicted = []
    error = 0
    with torch.no_grad():
        for x, y, z in test_loader_unshuffled:
            x = x.to(device)
            y = np.array(y.numpy())
            z = z.to(device)

            new_input = torch.empty((batch_size, window_size, dimension)).to(device)
            z = torch.flatten(z, start_dim=1)
            for j in range(window_size):
                new_input[:, j, :] = torch.cat((x[:, j, :], z), dim=1)

            output = model(new_input).cpu().detach().numpy()

            error += (output - y) ** 2

            predicted.extend(output)
            ground_truths.extend(y)

    run_error = error / len(test_loader_unshuffled)
    RMSE = np.sqrt(np.mean(run_error))

    predicted = np.asarray(predicted)
    ground_truths = np.asarray(ground_truths)
    for i in range(len(fingers)):
        plt.subplot(1, len(fingers), i + 1)
        plt.plot(predicted[:, i], color='red', label='TCN Prediction')
        plt.plot(ground_truths[:, i], color='blue', label='Ground Truth')
        plt.xlabel('Time (ms)')
        plt.ylabel('Percent Finger Flexion of %s' % finger_strings[i])
        plt.title('TCN vs Ground Truth')
        plt.legend()
    plt.show()

    score_metric = 1 - ((np.sum((predicted - ground_truths) ** 2)) / (np.sum((ground_truths - np.mean(ground_truths)) ** 2)))

    logger.info("Test RMSE {:.4f} | Goodness of fit {:.4f} ".format(RMSE, score_metric.item()))
