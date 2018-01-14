import sys
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model import Encoder, Attn, AttnDecoder
from load import LanguageDataset,MAX_LENGTH, VOCAB_SIZE
from utils import *
from tqdm import tqdm
import os

#TODO: EDIT VALIDATION PROCESS
'''
def valid(dataset, en, de, recon_fn, out_prefix, save_prefix,nE):
    outfile = open('{}_{}.csv'.format(out_prefix,nE), 'w')
    val_logfile = open(save_prefix+'_valid_logger.csv', 'a')

    vMSE = 0
    nVal = 0
    en.eval()
    de.eval()
    dataset.eval()
    eStart = time.time()
    print('Validation...')
    for part in dataset:
        pMSE = 0
        tStart = time.time()
        for nB, (batch_x, batch_l, batch_y) in enumerate(part):
            seq_len = 58
            batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
            batch_x = batch_x[indices]
            batch_y = batch_y[indices]
            batch_x = Variable(batch_x).cuda() if USE_CUDA else Variable(batch_x)
            packed_x = pack_padded_sequence(batch_x, batch_l.numpy(), batch_first=True)

            hidden = en.init_hidden(batch_x.size()[0])
            en_hidden = en(packed_x, hidden)

            de_input = Variable(torch.zeros(batch_x.size()[0],seq_len,39))
            if USE_CUDA:
                de_input = de_input.cuda()
            de_output, de_hidden = de(de_input,en_hidden)

            loss = maskMSE(de_output, batch_x, batch_l, recon_fn)

            pMSE += loss.data[0]

            for i, feat in enumerate(de_hidden[0].data.squeeze(0).cpu().numpy()):
                print(batch_y[i], end=',', file=outfile)
                print(*feat, sep=' ', file=outfile)

        tEnd = time.time()
        print('Part %s | loss %6.3f | cost %s'
              % (dataset.part, pMSE/len(part.dataset), tfmt(tEnd-tStart)))

        vMSE += pMSE
        nVal += len(part.dataset)

    eEnd = time.time()
    print('validation loss %6.3f , cost %s\n' % ((vMSE/nVal), tfmt(eEnd-eStart)))
    print(vMSE/nVal, file=val_logfile)

    return vMSE
'''

log_softmax = nn.LogSoftmax()

def maskCCE(output, target, mask):
    total = mask.sum()
    crossEntropy = -torch.gather(log_softmax(output), 1, target.contiguous().view(-1, 1))
    loss = crossEntropy.masked_select(mask.view(-1, 1)).mean()
    # loss = loss.cuda() if USE_CUDA else loss
    return loss,total.data[0]

def maskCCE_new(output, target, mask):
    total = mask.sum()
    crossEntropy = -torch.gather(log_softmax(output.view(-1, VOCAB_SIZE)), 1, target.contiguous().view(-1, 1))
    loss = crossEntropy.masked_select(mask.view(-1, 1)).mean()
    # loss = loss.cuda() if USE_CUDA else loss
    return loss,total.data[0]

def len2mask(batch_l):
    mask = torch.zeros(len(batch_l),max(batch_l))
    for i in range(len(batch_l)):
        for j  in range(max(batch_l)):
            if j < batch_l[i]:
                mask[i][j] = 1
    if USE_CUDA:
        mask = mask.cuda()
    return Variable(mask.byte())

def Train(verbose, l1, l2, iteration, lr, batch_size, hidden_size, vocab_size, print_every, save_every, postfix, save_dir):

    print("Building Encoder and Decoder...")
    input_size = 300
    output_size = vocab_size
    en = Encoder(input_size, hidden_size)
    l1_de = AttnDecoder('general', input_size, hidden_size, output_size)
    l2_de = AttnDecoder('general', input_size, hidden_size, output_size)

    if USE_CUDA:
        en = en.cuda()
        l1_de = l1_de.cuda()
        l2_de = l2_de.cuda()

    en_opt = optim.Adam(en.parameters(),lr = 0.0002)
    l1_de_opt = optim.Adam(l1_de.parameters(), lr = 0.0002)
    l2_de_opt = optim.Adam(l2_de.parameters(), lr = 0.0002)
    print("Creating l1 dataset...")
    l1_dataset = LanguageDataset('data/data_{}.subword.clean'.format(l1),l1,verbose)
    print("Creating l2 dataset...")
    l2_dataset = LanguageDataset('data/data_{}.subword.clean'.format(l2),l2,verbose)

    l1_loader = DataLoader(l1_dataset, batch_size = batch_size, shuffle = True)
    l2_loader = DataLoader(l2_dataset, batch_size = batch_size, shuffle = True)

    def infinite_loader(iterable):
        while True:
            for batch in iterable:
                yield batch

    l1_bigdata = infinite_loader(l1_loader)
    l2_bigdata = infinite_loader(l2_loader)

    de_l1_loss, de_l2_loss, bt_l1_loss, bt_l2_loss = 0, 0, 0, 0


    print('Start training...')
    for it in range(iteration):
        # print("Denoising Language 1...")
        batch_index, batch_l, batch_sentence = next(l1_bigdata)
        bsz = len(batch_l) #Last batch may have different batch size
        if USE_CUDA:
            batch_index = batch_index.cuda()
            batch_l = batch_l.cuda()
            batch_sentence = batch_sentence.cuda()

        batch_sentence = Variable(sentence_swap(batch_l, batch_sentence))
        batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
        batch_sentence = batch_sentence[indices]
        batch_index = Variable(batch_index[indices])
        packed_sentence = pack_padded_sequence(batch_sentence, batch_l.cpu().numpy(), batch_first=True)

        en_opt.zero_grad()
        l1_de_opt.zero_grad()
        hidden = en.init_hidden(batch_sentence.size(0))
        packed_output,en_hidden = en(packed_sentence, hidden)
        output,_ = pad_packed_sequence(packed_output,batch_first = True)
        de_input = Variable(torch.zeros(bsz,1,input_size)) # Iteration through time steps
        de_context = Variable(torch.zeros(bsz,1,2*hidden_size))
        if USE_CUDA:
            de_input = de_input.cuda()
            de_context = de_context.cuda()
        de_hidden_0 = (en_hidden[0]+en_hidden[1]).unsqueeze(0)
        de_hidden_1 = (en_hidden[2]+en_hidden[3]).unsqueeze(0)
        de_hidden = torch.cat((de_hidden_0,de_hidden_1),0)

        denoising_loss_1 = 0
        #Total = 0
        mask = len2mask(batch_l)

        for di in range(output.size(1)):
            de_output, de_context, de_hidden, de_attention = l1_de(de_input, de_context, de_hidden, output)
            _,choose = torch.max(de_output,1)
            de_input = [l1_dataset.get_embed(index.data[0]).tolist() for index in choose]
            de_input = Variable(torch.FloatTensor(de_input).unsqueeze(1))
            if USE_CUDA:
                de_input = de_input.cuda()
            loss, total = maskCCE(de_output, batch_index[:,di], mask[:,di])
            denoising_loss_1 += loss
            #Total += total

        denoising_loss_1.backward()
        '''
        clip = 50.0
        torch.nn.utils.clip_grad_norm(en.parameters(), clip)
        torch.nn.utils.clip_grad_norm(de.parameters(), clip)
        '''
        en_opt.step()
        l1_de_opt.step()

        de_l1_loss += denoising_loss_1.data[0]/bsz
        # if (it+1) % print_every == 0:
        #     print('Loss for epoch {}: {}'.format(it, Epoch_loss/print_every))

        # print("Denoising Language 2...")
        batch_index, batch_l, batch_sentence = next(l2_bigdata)
        start = time.time()
        bsz = len(batch_l) #Last batch may have different batch size
        if USE_CUDA:
            batch_index = batch_index.cuda()
            batch_l = batch_l.cuda()
            batch_sentence = batch_sentence.cuda()

        batch_sentence = Variable(sentence_swap(batch_l, batch_sentence))
        batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
        batch_sentence = batch_sentence[indices]
        batch_index = Variable(batch_index[indices])
        packed_sentence = pack_padded_sequence(batch_sentence, batch_l.cpu().numpy(), batch_first=True)

        current = time.time()

        en.zero_grad()
        l2_de_opt.zero_grad()
        hidden = en.init_hidden(batch_sentence.size(0))
        packed_output,en_hidden = en(packed_sentence, hidden)
        output,_ = pad_packed_sequence(packed_output,batch_first = True)
        de_input = Variable(torch.zeros(bsz,1,input_size)) # Iteration through time steps
        de_context = Variable(torch.zeros(bsz,1,2*hidden_size))
        if USE_CUDA:
            de_input = de_input.cuda()
            de_context = de_context.cuda()
        de_hidden_0 = (en_hidden[0]+en_hidden[1]).unsqueeze(0)
        de_hidden_1 = (en_hidden[2]+en_hidden[3]).unsqueeze(0)
        de_hidden = torch.cat((de_hidden_0,de_hidden_1),0)

        current = time.time()

        denoising_loss_2 = 0
        #Total = 0
        mask = len2mask(batch_l)

        for di in range(output.size(1)):
            de_output, de_context, de_hidden, de_attention = l2_de(de_input, de_context, de_hidden, output)
            _,choose = torch.max(de_output,1)
            de_input = [l2_dataset.get_embed(index.data[0]).tolist() for index in choose]
            de_input = Variable(torch.FloatTensor(de_input).unsqueeze(1))
            if USE_CUDA:
                de_input = de_input.cuda()
            loss, total = maskCCE(de_output, batch_index[:,di], mask[:,di])
            denoising_loss_2 += loss
            #Total += total

        denoising_loss_2.backward()
        '''
        clip = 50.0
        torch.nn.utils.clip_grad_norm(en.parameters(), clip)
        torch.nn.utils.clip_grad_norm(de.parameters(), clip)
        '''
        en_opt.step()
        l2_de_opt.step()
#         print(denoising_loss_1.data[0]/bsz)
        de_l2_loss += denoising_loss_2.data[0]/bsz

        # if (it+1) % print_every == 0:
        #     print('Loss for iteration {}: {}'.format(it, Epoch_loss/print_every))

        # print("Backtranslation on Language 1...")
        batch_index, batch_l, batch_sentence = next(l1_bigdata)
        bsz = len(batch_l) #Last batch may have different batch size
        if USE_CUDA:
            batch_index = batch_index.cuda()
            batch_l = batch_l.cuda()
            batch_sentence = batch_sentence.cuda()

        batch_sentence = Variable(batch_sentence)
        batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
        batch_sentence = batch_sentence[indices]
        batch_index = Variable(batch_index[indices])
        packed_sentence = pack_padded_sequence(batch_sentence, batch_l.cpu().numpy(), batch_first=True)

        l2_de.eval()

        en_opt.zero_grad()
        l1_de_opt.zero_grad()

        l2_hidden = en.init_hidden(batch_sentence.size(0))
        packed_output,en_l2_hidden = en(packed_sentence, l2_hidden)
        output,_ = pad_packed_sequence(packed_output,batch_first = True)
        de_l2_input = Variable(torch.zeros(bsz,1,input_size)) 
        de_l2_context = Variable(torch.zeros(bsz,1,2*hidden_size))
        if USE_CUDA:
            de_l2_input = de_l2_input.cuda()
            de_l2_context = de_l2_context.cuda()
        de_l2_hidden_0 = (en_l2_hidden[0]+en_l2_hidden[1]).unsqueeze(0)
        de_l2_hidden_1 = (en_l2_hidden[2]+en_l2_hidden[3]).unsqueeze(0)
        de_l2_hidden = torch.cat((de_l2_hidden_0,de_l2_hidden_1),0)

        de_l2_output = Variable(torch.zeros(bsz, MAX_LENGTH, 1, input_size)).cuda() if USE_CUDA else Variable(torch.zeros(bsz, MAX_LENGTH, 1, input_size))


        # Sequential output
        for di in range(output.size(1)):
            if di == 0:
                uni_output, de_l2_context, de_l2_hidden, de_l2_attention = l2_de(de_l2_input, de_l2_context, de_l2_hidden, output)
            else:
                uni_output, de_l2_context, de_l2_hidden, de_l2_attention = l2_de(de_l2_output[:,di-1,:,:], de_l2_context, de_l2_hidden, output)
            _,choose = torch.max(uni_output,1)
            de_l2_input = [l2_dataset.get_embed(index.data[0]).tolist() for index in choose]
            de_l2_input = torch.FloatTensor(de_l2_input).unsqueeze(1)
            de_l2_output[:,di,:,:] = de_l2_input


        de_l2_output = de_l2_output.squeeze(2) # Remove the extra dimension
        ##de_l2_output is the (fake) translation
        ##TODO: Check if de_l2_output is a valid input for encoder 
        l1_hidden = en.init_hidden(batch_sentence.size(0))
        packed_output,en_l1_hidden = en(de_l2_output, l1_hidden)
#        output,_ = pad_packed_sequence(packed_output,batch_first = True)
        de_l1_input = Variable(torch.zeros(bsz,1,input_size)) 
        de_l1_context = Variable(torch.zeros(bsz,1,2*hidden_size))
        if USE_CUDA:
            de_l1_input = de_l1_input.cuda()
            de_l1_context = de_l1_context.cuda()
        de_l1_hidden_0 = (en_l1_hidden[0]+en_l1_hidden[1]).unsqueeze(0)
        de_l1_hidden_1 = (en_l1_hidden[2]+en_l1_hidden[3]).unsqueeze(0)
        de_l1_hidden = torch.cat((de_l1_hidden_0,de_l1_hidden_1),0)

        mask = len2mask(batch_l)

        de_l1_long_output = Variable(torch.zeros(bsz, MAX_LENGTH, VOCAB_SIZE))
        if USE_CUDA:
            de_l1_long_output = de_l1_long_output.cuda()

        for di in range(output.size(1)):
            de_l1_output, de_l1_context, de_l1_hidden, de_l1_attention = l1_de(de_l1_input, de_l1_context, de_l1_hidden, output)
            de_l1_long_output[:,di,:] = de_l1_output
            _,choose = torch.max(de_l1_output,1)
            de_l1_input = [l1_dataset.get_embed(index.data[0]).tolist() for index in choose]
            de_l1_input = Variable(torch.FloatTensor(de_l1_input).unsqueeze(1))
            if USE_CUDA:
                de_l1_input = de_l1_input.cuda()

        backtranslation_loss_1, total = maskCCE_new(de_l1_long_output, batch_index, mask)
        backtranslation_loss_1.backward()

        en_opt.step()
        l1_de_opt.step()
        bt_l1_loss += backtranslation_loss_1.data[0] / bsz

        # print("Backtranslation on Language 2...")
        batch_index, batch_l, batch_sentence = next(l2_bigdata)
        bsz = len(batch_l) #Last batch may have different batch size
        if USE_CUDA:
            batch_index = batch_index.cuda()
            batch_l = batch_l.cuda()
            batch_sentence = batch_sentence.cuda()

        batch_sentence = Variable(batch_sentence)
        batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
        batch_sentence = batch_sentence[indices]
        batch_index = Variable(batch_index[indices])
        packed_sentence = pack_padded_sequence(batch_sentence, batch_l.cpu().numpy(), batch_first=True)
        
        l1_de.eval()
        
        en_opt.zero_grad()
        l2_de_opt.zero_grad()
       
        l1_hidden = en.init_hidden(batch_sentence.size(0))
        packed_output,en_l1_hidden = en(packed_sentence, l1_hidden)
        output,_ = pad_packed_sequence(packed_output,batch_first = True)
        de_l1_input = Variable(torch.zeros(bsz,1,input_size)) 
        de_l1_context = Variable(torch.zeros(bsz,1,2*hidden_size))
        if USE_CUDA:
            de_l1_input = de_l1_input.cuda()
            de_l1_context = de_l1_context.cuda()
        de_l1_hidden_0 = (en_l1_hidden[0]+en_l1_hidden[1]).unsqueeze(0)
        de_l1_hidden_1 = (en_l1_hidden[2]+en_l1_hidden[3]).unsqueeze(0)
        de_l1_hidden = torch.cat((de_l1_hidden_0,de_l1_hidden_1),0)

        de_l1_output = Variable(torch.zeros(bsz, MAX_LENGTH, 1, input_size)).cuda() if USE_CUDA else Variable(torch.zeros(bsz, MAX_LENGTH, 1, input_size))
        
        for di in range(output.size(1)):
            if di == 0:
                uni_output, de_l1_context, de_h1_hidden, de_l1_attention = l1_de(de_l1_input, de_l1_context, de_l1_hidden, output)
            else:
                uni_output, de_l1_context, de_h1_hidden, de_l1_attention = l1_de(de_l1_output[:,di-1,:,:], de_l1_context, de_l1_hidden, output)

            _,choose = torch.max(uni_output,1)
            de_l1_input = [l1_dataset.get_embed(index.data[0]).tolist() for index in choose]
            de_l1_input = torch.FloatTensor(de_l1_input).unsqueeze(1)
            de_l1_output[:,di,:,:] = de_l1_input
        
        de_l1_output = de_l1_output.squeeze(2)
        ##de_l2_output is the (fake) translation
        ##TODO: Check if de_l2_output is a valid input for encoder 
        l2_hidden = en.init_hidden(batch_sentence.size(0))
        packed_output, en_l2_hidden = en(de_l1_output, l2_hidden)
        de_l2_input = Variable(torch.zeros(bsz,1,input_size)) 
        de_l2_context = Variable(torch.zeros(bsz,1,2*hidden_size))
        if USE_CUDA:
            de_l2_input = de_l2_input.cuda()
            de_l2_context = de_l2_context.cuda()
        de_l2_hidden_0 = (en_l2_hidden[0]+en_l2_hidden[1]).unsqueeze(0)
        de_l2_hidden_1 = (en_l2_hidden[2]+en_l2_hidden[3]).unsqueeze(0)
        de_l2_hidden = torch.cat((de_l2_hidden_0,de_l2_hidden_1),0)

        mask = len2mask(batch_l)

        de_l2_long_output = Variable(torch.zeros(bsz, MAX_LENGTH, VOCAB_SIZE))
        if USE_CUDA:
            de_l2_long_output = de_l2_long_output.cuda()

        for di in range(output.size(1)):
            de_l2_output, de_l2_context, de_l2_hidden, de_l2_attention = l2_de(de_l2_input, de_l2_context, de_l2_hidden, output)
            de_l2_long_output[:,di,:] = de_l2_output
            _,choose = torch.max(de_l2_output,1)
            de_l2_input = [l2_dataset.get_embed(index.data[0]).tolist() for index in choose]
            de_l2_input = Variable(torch.FloatTensor(de_l2_input).unsqueeze(1))
            if USE_CUDA:
                de_l2_input = de_l2_input.cuda()

        backtranslation_loss_2, total = maskCCE_new(de_l2_long_output, batch_index, mask)
        backtranslation_loss_2.backward()

        en_opt.step()
        l2_de_opt.step()

        bt_l2_loss += backtranslation_loss_2.data[0] / bsz
        


        if it % print_every == 0:
            print('Iter. {}, de_l1_loss = {}, de_l2_loss={}, bt_l1_loss = {}, bt_l2_loss={}'.format(it, de_l1_loss, de_l2_loss, bt_l1_loss, bt_l2_loss))
            with open(os.path.join(save_dir, 'log_' + postfix + '.csv'), 'a') as log_file:
                log_file.write('{},{},{},{},{}'.format(it, de_l1_loss, de_l2_loss, bt_l1_loss, bt_l2_loss) + '\n')
            de_l1_loss, de_l2_loss, bt_l1_loss, bt_l2_loss = 0, 0, 0, 0
            
        if it % save_every == 0:
            torch.save({
                'en':en.state_dict(),
                'l1_de':l1_de.state_dict(),
                'l2_de':l2_de.state_dict(),
                'en_opt':en_opt.state_dict(),
                'l1_de_opt':l1_de_opt.state_dict(),
                'l2_de_opt':l2_de_opt.state_dict()},
                postfix+'e_{}.pkl'.format(it))
            
#        print('Time for each iteration:', time.time()-start)




