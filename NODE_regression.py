import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
import os

###############
#Added so it can run from file
import data_utils
import sys
sys.path.insert(0, '/Users/ScottEnsel/torchdiffeq')
###############


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save', type=str, default='./experiment')

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class Full_Linear(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True,):
        super(Full_Linear, self).__init__()
        module = nn.Linear
        self._layer = module(dim_in + 1, dim_out, bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()

        self.norm1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim*3)
        self.fc3 = nn.Linear(dim*3, dim*2)
        self.fc4 = nn.Linear(dim*2, dim)
        self.fc5 = nn.Linear(dim, dim)

        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        out = self.norm1(x)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.fc5(out)
        out = self.relu(out)

        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        # can specify method
        '''SOLVERS = {
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        }
        '''
        #None = dopri5
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method=None)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

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
    for x, y in dataset_loader:
        x = x.to(device)
        y = np.array(y.numpy())

        predicted = model(x).cpu().detach().numpy()

        error += (predicted - y)**2

    run_error = error / len(dataset_loader)

    RMSE = np.sqrt(np.mean(run_error))

    return RMSE

def smooth_loss(C, predictions, reg):

    loss = torch.sum((C*predictions)**2)

    loss = reg*loss

    return loss

if __name__ == '__main__':

    args.save = './Testingtest'

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    THUMB_INDEX = 0
    INDEX_INDEX = 1
    MIDDLE_INDEX = 2
    RING_INDEX = 3
    PINKY_INDEX = 4
    finger_strings = ["Thumb","Index","Middle","Ring","Pinky"]

    args.nepochs = 1
    batch_size = 512 #was 50
    args.batch_size = batch_size
    window_size = 10
    num_labels = 3
    fingers = [THUMB_INDEX, INDEX_INDEX, MIDDLE_INDEX]
    # reg = 0.001

    which_feats = [0, 1, 2, 3, 4, 5, 6, 7]

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    #creat C matrix
    #put this in where you define everything else so it isnt created everytime you run the function
    # C = np.zeros((batch_size, batch_size), int)
    # np.fill_diagonal(C, -1)
    # np.fill_diagonal(C[:, 1:], 1)
    # C[-1, -1] = 0
    # C = torch.from_numpy(C).float().to(device)

    all_data = sio.loadmat('/Users/ScottEnsel/Desktop/Deep Learning/Project/NEW files/Z_run-010_thumb_index_middle.mat',
                           struct_as_record=False, squeeze_me=True)
    EMG_data = all_data['z']

    #load in our data
    # all_data = sio.loadmat(os.path.join(data_utils.DATA_DIR,data_utils.DATA_SET1), struct_as_record=False, squeeze_me=True)
    # EMG_data = all_data['z']


    train_loader, test_loader, valid_loader = data_utils.get_data_loaders(EMG_data, fingers, num_labels,
                                                                          which_feats, window_size, batch_size,
                                                                          train_split=0.8, validation_split=0.2, center=False)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    dimension = len(which_feats) + ((window_size - 1)*len(fingers))

    feature_layers = [ODEBlock(ODEfunc(dimension))]
    fc_layers = [nn.Linear(dimension, len(fingers))]

    model = nn.Sequential(*feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=batch_size, batches_per_epoch=batches_per_epoch, boundary_epochs=[2, 6, 12, 18],
        decay_rates=[1, 0.1, 0.01, 0.001, 0.0001]
    )

    best_err = float("inf")
    for itr in range(args.nepochs * batches_per_epoch):

        if itr % batches_per_epoch == 0:
            print("Epoch %.4g" % (itr//batches_per_epoch))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()

        x = x.to(device)
        y = y.to(device)

        predictions = model(x)

        loss = torch.mean(torch.abs(predictions - y))

        nfe_forward = feature_layers[0].nfe
        feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        nfe_backward = feature_layers[0].nfe
        feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)

        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)

        end = time.time()

        if itr % batches_per_epoch == (batches_per_epoch-1):
            with torch.no_grad():
                train_err = RMSE_error(model, train_loader)
                val_err = RMSE_error(model, valid_loader)

                if val_err < best_err:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_err = val_err

                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Validation Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_err, val_err
                    )
                )

            ground_truths = []
            predicted = []
            for x, y in test_loader:
                x = x.to(device)
                y = np.array(y.numpy())

                output = model(x).cpu().detach().numpy()

                predicted.extend(output)
                ground_truths.extend(y)

            predicted = np.asarray(predicted)
            ground_truths = np.asarray(ground_truths)
            for i in range(len(fingers)):
                plt.plot(predicted[:, i], color='red', label='NODE Prediction')
                plt.plot(ground_truths[:, i], color='blue', label='Ground Truth')
                plt.xlabel('Time (ms)')
                plt.ylabel('Percent Finger Flexion of %s' % finger_strings[i])
                plt.title('NODE vs Ground Truth Test %.4d' % (itr // batches_per_epoch))
                plt.legend()
                plt.show()

    with torch.no_grad():
        test_err = RMSE_error(model, test_loader)

        ground_truths = []
        predicted = []
        for x, y in test_loader:
            x = x.to(device)
            y = np.array(y.numpy())

            output = model(x).cpu().detach().numpy()

            predicted.extend(output)
            ground_truths.extend(y)

        predicted = np.asarray(predicted)
        ground_truths = np.asarray(ground_truths)

        score_metric = 1 - ((np.sum((predicted - ground_truths) ** 2)) / (np.sum((ground_truths - np.mean(ground_truths))** 2)))

        logger.info("Test Err {:.4f} | Goodness of fit {:.4f} ".format(test_err, score_metric.item()))

        for i in range(len(fingers)):
            plt.plot(predicted[:,i], color='red', label='NODE Prediction')
            plt.plot(ground_truths[:,i], color='blue', label='Ground Truth')
            plt.xlabel('Time (ms)')

            plt.ylabel('Percent Finger Flexion of %s' %finger_strings[i])
            plt.title('NODE vs Ground Truth Test')
            plt.legend()
            plt.show()

