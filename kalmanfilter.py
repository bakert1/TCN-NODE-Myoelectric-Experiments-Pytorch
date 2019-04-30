import numpy as np
import scipy.io as sio
from numpy.linalg import inv
import matplotlib.pyplot as plt
import data_utils
import os
from sklearn.model_selection import train_test_split
import sys
from numpy.linalg import inv


def data_split(X, Y, split = 0.10):
    if split > 0.5:
        print("Error: split too large. Reduce below 0.5")
        sys.exit(1)
    else:

        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = split, random_state=0, shuffle=False)

        #TODO: We normalize (z_score) the inputs and zero-center the outputs.
        # Parameters for z-scoring (mean/std.) should be determined on the training set only,
        # and then these z-scoring parameters are also used on the testing and validation sets

        # Z-score inputs 
        X_kf_train_mean = np.nanmean(X_train, axis=0)
        X_kf_train_std = np.nanstd(X_train, axis=0)
        X_train = (X_train - X_kf_train_mean) / X_kf_train_std
        X_test = (X_test - X_kf_train_mean) / X_kf_train_std

        # Zero-center outputs
        y_kf_train_mean = np.mean(y_train, axis=0)
        y_train = y_train - y_kf_train_mean
        y_test = y_test - y_kf_train_mean

        # add a column of ones to data
        n, _ = X_train.shape
        X0 = np.ones((n, 1))
        X_train = np.hstack((X0,X_train))

        m, _ = X_test.shape
        X0 = np.ones((m, 1))
        X_test = np.hstack((X0,X_test))

        # this is so shape of y is (_,1) and not (_,)
        #fix for more than one moment
        #TODO: fix this so it doesnt only work for no dimension
        y_train = np.reshape(y_train,(len(y_train),1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        #transpose all the matrices so the kalman filter math works out
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T

        return X_train, X_test, y_train, y_test, y_kf_train_mean

def Kalman_filter(x_train, x_test, y_train, y_test):
    j, k = x_train.shape
    n, m = y_test.shape

    A = np.dot(np.dot(y_train[:,1:], y_train[:,0:-1].T) , inv(np.dot(y_train[:,0:-1],y_train[:,0:-1].T)))
    C = np.dot(np.dot(x_train[:,1:], y_train[:,1:].T) , inv(np.dot(y_train[:,1:],y_train[:,1:].T)))

    W = (1/(y_train.shape[1]-1)) * np.dot((y_train[:,1:] - np.dot(A, y_train[:,0:-1])), (y_train[:,1:] - np.dot(A, y_train[:,0:-1])).T)
    Q = (1/(y_train.shape[1])) * np.dot((x_train[:,1:] - np.dot(C, y_train[:,1:])) , (x_train[:,1:] - np.dot(C , y_train[:,1:])).T)

    Pt = np.copy(W)
    x_hat2 = np.copy(y_train[:n,:m])
    x_hat = np.zeros((x_hat2.shape))

    for i in range(1,y_test.shape[1]-1):
        x_hat[:,i] = np.dot(A ,x_hat2[:,i]) #n,1 matrix
        Pt_t1 = np.dot(np.dot(A , Pt) , A.T) + W #n,n matrix
        #print(inv(np.dot(np.dot(C , Pt_t1) , C.T) + Q))
        Kt = np.dot(np.dot(Pt_t1 , C.T) , inv(np.dot(np.dot(C , Pt_t1) , C.T) + Q)) #n,j matrix
        x_hat2[:,i+1] = x_hat[:,i] + np.dot(Kt, (x_test[:,i+1] - np.dot(C , x_hat[:,i])) )  #j,1 matrix
        #x_hat2[:,i+1] = x_hat[:,i] + np.dot(Kt, (np.reshape(x_test[:,i+1],(j,1)) - np.reshape(np.dot(C , x_hat[:,i]),(j,n)) ) ) #j,1 matrix
        Pt = np.dot((1 - np.dot(Kt , C)) , Pt_t1)

    return x_hat2

def fit_loss_check(predictions, true_labels, training_mean):

    fit = 1 - ((np.sum(predictions - true_labels)**2))/(np.sum(true_labels - training_mean)**2)

    return fit

def RMSE_error(predictions, labels): # should be named error

    error = predictions - labels
    RMSE = np.sqrt(np.mean(error**2))

    return RMSE

if __name__ == '__main__':

    all_data = sio.loadmat('/Users/ScottEnsel/Desktop/Deep Learning/Project/NEW files/Z_run-010_thumb_index_middle.mat',
                           struct_as_record=False, squeeze_me=True)
    EMG_data = all_data['z']
    #
    # all_data = sio.loadmat(os.path.join(data_utils.DATA_DIR,data_utils.DATA_SET1), struct_as_record=False, squeeze_me=True)
    # EMG_data = all_data['z']

    THUMB_INDEX = 0
    INDEX_INDEX = 1
    MIDDLE_INDEX = 2
    RING_INDEX = 3
    PINKY_INDEX = 4

    # new_z = data_utils.preprocess_data(EMG_data, THUMB_INDEX)
    new_z = data_utils.preprocess_data(EMG_data, INDEX_INDEX)
    # new_z = data_utils.preprocess_data(EMG_data, MIDDLE_INDEX)

    y = new_z[:,0] #seperate labels
    x = new_z[:,1:] #seperate features
    #1:34

    #split must be less than 0.5
    x_train, x_test, y_train, y_test, y_kf_train_mean = data_split(x, y, split=0.04)

    training_mean = np.sum(y_train)

    x_hat = Kalman_filter(x_train, x_test, y_train, y_test)

    error = x_hat - y_test

    RMSE = RMSE_error(x_hat, y_test)
    print("Test Error %4.4f" %RMSE)

    fit = fit_loss_check(x_hat, y_test, training_mean)
    print("Goodness of Fit %4.4f" %fit)

    line1 = plt.plot(y_test.T + y_kf_train_mean, color='Blue', label='Ground Truth')
    line2 = plt.plot(x_hat.T + y_kf_train_mean, color='Red', label='Kalman Filter')
    plt.xlabel('Time (ms)')
    plt.ylabel('Percent Finger Flexion')
    plt.title('Kalman Filter Prediction vs Ground Truth')
    #plt.ylim(0, 1)
    plt.legend()
    plt.show()
