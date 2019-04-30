from torch import nn
from TCN.tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, window_size, num_fingers, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear((8+(window_size-1)*num_fingers)*num_channels[-1], (8+(window_size-1)*num_fingers)*num_channels[-1])
        self.linear2 = nn.Linear((8+(window_size-1)*num_fingers)*num_channels[-1], output_size)


    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        

    def forward(self, x):
        batch_size = x.size()[0]
        y1 = self.tcn(x)
        y1 = y1.reshape(batch_size, -1)
        y2 = self.linear1(y1)
        output = self.linear2(y1)
        return output