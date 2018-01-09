import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import USE_CUDA
from load import MAX_LENGTH


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2,
                          bidirectional=True, batch_first=True)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(4, batch_size, self.hidden_size))
        if USE_CUDA:
            return hidden.cuda()
        else:
            return hidden

    def forward(self, input_seq, hidden):
        outputs, hidden = self.gru(input_seq, hidden)  # output: ( batch,seq_len, hidden*n_dir)
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, seq_len))  # B x S
        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[:, i] = self.score(hidden, encoder_outputs[:, i, :])
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = Variable(torch.zeros(encoder_output.size(0)))
            tmp = self.attn(encoder_output)
            if USE_CUDA:
                energy = energy.cuda()
            for idx in range(hidden.size(0)):
                energy[idx] = hidden[idx].dot(tmp[idx])

            return energy


class AttnDecoder(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size):
        super(AttnDecoder, self).__init__()
        self.attn_model = attn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru_input_size = self.input_size + self.hidden_size * 2
        if attn_model != None:
            self.attn = Attn(attn_model, hidden_size)
        self.gru = nn.GRU(self.gru_input_size, self.hidden_size, num_layers=2, batch_first=True)
        self.out = nn.Linear(self.hidden_size * 3, self.output_size)

    def forward(self, input, context, hidden, encoder_outputs):
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((input, context), 2)
        rnn_output, hidden = self.gru(rnn_input, hidden) # B x 1 x H, 2 x B x H
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(1), encoder_outputs)

        context = attn_weights.bmm(encoder_outputs)  # B x 1 x 2H

        # Final output layer (next word prediction) using the RNN hidden state and context vector

        #output = F.softmax(self.out(torch.cat((rnn_output.squeeze(1), context.squeeze(1)), 1)))
        output = self.out(torch.cat((rnn_output.squeeze(1), context.squeeze(1)), 1))


        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if USE_CUDA:
            return hidden.cuda()
        else:
            return hidden


def main():
    batch_size = 5
    seq_length = 10
    input_size = 20
    hidden_size = 30
    output_size = 100
    encoder = Encoder(input_size, hidden_size)
    decoder = AttnDecoder('general', input_size, hidden_size, output_size)
    test_seq = Variable(torch.rand(batch_size, seq_length, input_size))
    test_word = Variable(torch.rand(batch_size, 1, input_size)) # B x 1 x H
    test_context = Variable(torch.zeros(batch_size, 1, hidden_size * 2)) # B x 1 x 2H
    encoder_hidden = encoder.init_hidden(batch_size)
    output, hidden = encoder(test_seq, encoder_hidden)
    output, context, hidden, attn_weights = decoder(test_word, test_context, hidden, output)
    print(output)


if __name__ == '__main__':

    main()
