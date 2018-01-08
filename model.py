import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import USE_CUDA
from load import MAX_LENGTH

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.RNN = nn.GRU(input_size, hidden_size, num_layers = 2, bidirectional = True, batch_first = True)
    
    def init_hidden(self,batch_size):
        hidden = Variable(torch.zeros(4, batch_size, self.hidden_size))
        if USE_CUDA:
            return hidden.cuda() 
        else:
            return hidden
 
    def forward(self, input_seq, hidden):
        outputs, hidden = self.RNN(input_seq, hidden) # output: ( batch,seq_len, hidden*n_dir)
        return outputs,hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        # Create variable to store attention energies
        encoder_outputs = encoder_outputs[:,:,:self.hidden_size] + encoder_outputs[:,:,self.hidden_size:]
        attn_energies = Variable(torch.zeros(seq_len,encoder_outputs.size(1),1)) # B x 1 x S
        if USE_CUDA: 
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
            print(attn_energies[i])
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = Variable(torch.zeros(encoder_output.size(0)))
            tmp = self.attn(encoder_output)
            if USE_CUDA:
                energy = energy.cuda()
            for idx,_ in enumerate(energy):
                energy[0] = hidden[idx].dot(tmp[idx])
            print(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy

class AttnDecoder(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        if attn_model != None:
            self.attn = Attn(attn_model, hidden_size)
        self.RNN = nn.GRU(self.hidden_size*2, self.hidden_size, num_layers = 2)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, context, hidden, encoder_outputs):
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((input, context.unsqueeze(0)),2)
        rnn_output, hidden = self.RNN(rnn_input, hidden)
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        print(attn_weights)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
    
    def init_hidden(self,batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            return hidden.cuda()
        else:
            return hidden
    
