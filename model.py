import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import USE_CUDA

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.RNN = nn.GRU(input_size, hidden_size, num_layers = 2, bidirectional = True)
    
    def init_hidden(self,batch_size):
        if USE_CUDA:
            return (Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()),
                    Variable(torch.zeros(1, batch_size, self.hidden_size).cuda()))
        else:
            return (Variable(torch.zeros(1, batch_size, self.hidden_size)),
                    Variable(torch.zeros(1, batch_size, self.hidden_size))) 
 
    def forward(self, input_seq, hidden):
        outputs, hidden = self.RNN(input_seq, hidden) # output: ( batch,seq_len, hidden*n_dir)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first = True)
        return hidden

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True)
        self.hidden_to_out = nn.Linear(hidden_size,input_size)
    
    def forward(self, input_seq, hidden):
        output, hidden = self.lstm(input_seq, hidden) #output: (batch_size, seq_len, hidden)
        batch_size = output.size()[0]
        output = self.hidden_to_out(output.contiguous().view(-1,self.hidden_size))
        output = output.view(batch_size,-1,39)
        return output, hidden
 
class VRAE(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(VRAE, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = nn.LSTM(input_size, 2*hidden_size, n_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.hidden_to_out = nn.Linear(hidden_size, input_size)
    
    def init_hidden(self, batch_size):
        if USE_CUDA:
            return (Variable(torch.zeros(self.n_layers, batch_size, 2*self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.n_layers, batch_size, 2*self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers, batch_size, 2*self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, 2*self.hidden_size)))

    def encode(self, x, hidden):
        _, hidden = self.encoder(x, hidden)
        mean = (hidden[0][:,:,:self.hidden_size], hidden[1][:,:,:self.hidden_size])
        logvar = (hidden[0][:,:,self.hidden_size:], hidden[1][:,:,self.hidden_size:])
        return mean, logvar

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if USE_CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def decode(self, z, batch_size, seq_len):
        if USE_CUDA:
            input = Variable(torch.zeros(batch_size, seq_len, self.input_size)).cuda()
        else:
            input = Variable(torch.zeros(batch_size, seq_len, self.input_size))
        output, _ = self.decoder(input, z)
        output = self.hidden_to_out(output.contiguous().view(-1, self.hidden_size))
        output = output.view(batch_size, -1, self.input_size)
        return output

    def forward(self, x, hidden, batch_size):
        seq_len = len(x.batch_sizes)
        mean, logvar = self.encode(x, hidden)
        z = (self.reparametrize(mean[0], logvar[0]),
             self.reparametrize(mean[1], logvar[1]))
        output = self.decode(z, batch_size, seq_len)
        return mean[0].squeeze(), logvar[0].squeeze(), z[0].squeeze(), output

def maskMSE(recon_x, x, lens, recon_fn):
    MSE = 0
    for i in range(x.size()[0]):
        MSE += recon_fn(recon_x[i][0:lens[i]], x[i][0:lens[i]])
    r
