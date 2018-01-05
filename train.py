import sys
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model import Encoder, Attn, AttnDecoder 
from load import LanguageDataset
from utils import *

#TODO: EDIT VALIDATION PROCESS
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

#TODO:EDIT TRAINING PROCESS
def Train(verbose, l1, l2, epoch, lr, batch_size, hidden_size, vocab_size, print_every, save_every, postfix, save_dir):

    print("Building Encoder and Decoder...")
    input_size = 100
    output_size = vocab_size
    en = Encoder(input_size, hidden_size)
    attn_model = Attn('general', hidden_size, 60) 
    l1_de = AttnDecoder(attn_model,hidden_size,output_size)
    l2_de = AttnDecoder(attn_model,hidden_size,output_size)

    if USE_CUDA:
        en = en.cuda()
        attn_model = attn_model.cuda()
        l1_de = l1_de.cuda()
        l2_de = l2_de.cuda()

    en_opt = optim.Adam(en.parameters(),lr = 0.0002)
    l1_de_opt = optim.Adam(l1_de.parameters(), lr = 0.0002)
    l2_de_opt = optim.Adam(l2_de.parameters(), lr = 0.0002)
    print("Creating l1 dataset...")
    l1_dataset = LanguageDataset('./data/data_{}.subword'.format(l1),'{}'.format(l1),verbose)
    print("Creating l2 dataset...")
    l2_dataset = LanguageDataset('./data/data_{}.subword'.format(l2),'{}'.format(l2),verbose)

    l1_loader = DataLoader(l1_dataset, batch_size = batch_size, shuffle = True)
    l2_loader = DataLoader(l2_dataset, batch_size = batch_size, shuffle = True)

    print('Start training...')
    for e in range(epoch):
        print("Denoising Language 1...")
        for batch_index, batch_l, batch_sentence in l1_loader:
            if USE_CUDA:
                batch_index = batch_index.cuda()
                batch_l = batch_l.cuda()
                batch_sentence = batch_sentence.cuda()
            #TODO: finish sentence_swap function in util.py
            batch_sentence = Variable(sentence_swap(batch_index,batch_sentence)[1])
            batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
            batch_sentence = batch_sentence[indices]
            packed_sentence = pack_padded_sequence(batch_sentence, batch_l.cpu().numpy(), batch_first=True)

            en_opt.zero_grad()
            l1_de_opt.zero_grad()
            hidden = en.init_hidden(batch_sentence.size(0))
            packed_output,en_hidden = en(packed_sentence, hidden)
            output,_ = pad_packed_sequence(packed_output,batch_first = True)
            ##################CHECK UNTIL HERE####################
            de_input = Variable(torch.zeros(1,MAX_LENGTH,hidden_size))
            de_context = Variable(torch.zeros(1,hidden_size))
            if USE_CUDA:
                de_input = de_input.cuda()
                de_context = de_context.cuda()
            de_hidden = en_hidden
            for di in range(MAX_LENGTH):
                de_output, de_context, de_hidden, de_attention = l1_de(de_input, de_context, de_hidden, output)
                choose,_ = torch.max(de_output)
                de_input = Variable(torch.FloatTensor(l1_dataset.emb[l1_dataset.vocab.index2word[choose]])
                if USE_CUDA:
                    de_input = de_input.cuda()
            loss = maskCE(de_output, batch_index, batch_l, recon_fn)
            loss.backward()
            '''
            clip = 50.0
            torch.nn.utils.clip_grad_norm(en.parameters(), clip)
            torch.nn.utils.clip_grad_norm(de.parameters(), clip)
            '''
            en_opt.step()
            l1_de_opt.step()
            
            pLoss += loss.data[0]

            if (nB+1) % 20 == 0:
                print('Epoch %3d | Part %s | Batch %3d | loss %6.3f' % (nE+1, dataset.part, nB+1, loss.data[0]/batch_x.size()[0]))
            #del batch_x, packed_x, hidden
            print('average loss %6.3f '
                  % (pLoss/len(part.dataset)))
            print(dataset.part,pLoss/len(part.dataset), sep=',', file=tr_logfile)
            '''
            if nP != len(dataset) - 1:
                print('')
            val_loss = valid(dataset, en, de, recon_fn,out_prefix, save_prefix,nE)
            torch.save({
                'en':en.state_dict(),
                'de':de.state_dict(),
                'en_opt':en_opt.state_dict(),
                'de_opt':de_opt.state_dict()},
                save_prefix+'_e%s.pkl' % str(nE+1))
            '''
        print("Denoising Language 2...")
        for batch_sentence, batch_l in enumerate(l2_loader):
            sen = Variable(sentence_swap(batch_sentence))
            batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
            batch_x = batch_x[indices] 
            batch_x = Variable(batch_x).cuda() if USE_CUDA else Variable(batch_x)
            packed_x = pack_padded_sequence(batch_x, batch_l.numpy(), batch_first=True)
                
            en_opt.zero_grad()
            l2_de_opt.zero_grad()
            hidden = en.init_hidden(batch_x.size(0))
            en_hidden = en(packed_x, hidden)
            de_input = Variable(torch.zeros(batch_x.size(0),seq_len,))
            if USE_CUDA:
                de_input = de_input.cuda()
            de_output, de_hidden = l2_de(de_input,en_hidden)

            loss = maskMSE(de_output, batch_x, batch_l, recon_fn)
            loss.backward()
            '''
            clip = 50.0
            torch.nn.utils.clip_grad_norm(en.parameters(), clip)
            torch.nn.utils.clip_grad_norm(de.parameters(), clip)
            '''
            en_opt.step()
            l2_de_opt.step()
            
        print("Backtranslation on Language 1...")
        for batch_sentence, batch_l in enumerate(l1_loader):
            l2_de.eval()
            batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
            batch_x = batch_x[indices] 
            batch_x = Variable(batch_x).cuda() if USE_CUDA else Variable(batch_x)
            packed_x = pack_padded_sequence(batch_x, batch_l.numpy(), batch_first=True)
                
            en_opt.zero_grad()
            l1_de_opt.zero_grad()

            l2_hidden = en.init_hidden(batch_x.size(0))
            en_l2_hidden = en(packed_x, l2_hidden)
            de_l2_input = Variable(torch.zeros(batch_x.size(0),seq_len,))
            if USE_CUDA:
                de_l2_input = de_l2_input.cuda()

            ##de_l2_output is the (fake) translation
            de_l2_output, de_l2_hidden = l2_de(de_l2_input,en_l2_hidden)

            l1_hidden = en.init_hidden(batch_x.size(0))
            en_l1_hidden = en(de_l2_output, l1_hidden)
            de_l1_input = Variable(torch.zeros(batch_x.size(0),seq_len,))
            if USE_CUDA:
                de_l1_input = de_l1_input.cuda()
            de_l1_output, de_l1_hidden = l1_de(de_l1_input,en_l1_hidden)

            loss = maskMSE(de_l1_output, batch_x, batch_l, recon_fn)
            loss.backward()
            
            en_opt.step()
            l1_de_opt.step()
        
        print("Backtranslation on Language 2...")
        for batch_sentence, batch_l in enumerate(l2_loader):
            l1_de.eval()
            batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
            batch_x = batch_x[indices] 
            batch_x = Variable(batch_x).cuda() if USE_CUDA else Variable(batch_x)
            packed_x = pack_padded_sequence(batch_x, batch_l.numpy(), batch_first=True)
                
            en_opt.zero_grad()
            l2_de_opt.zero_grad()

            l1_hidden = en.init_hidden(batch_x.size(0))
            en_l1_hidden = en(packed_x, l1_hidden)
            de_l1_input = Variable(torch.zeros(batch_x.size(0),seq_len,))
            if USE_CUDA:
                de_l1_input = de_l1_input.cuda()

            ##de_l2_output is the (fake) translation
            de_l1_output, de_l1_hidden = l1_de(de_l1_input,en_l1_hidden)

            l2_hidden = en.init_hidden(batch_x.size(0))
            en_l2_hidden = en(de_l1_output, l2_hidden)
            de_l2_input = Variable(torch.zeros(batch_x.size(0),seq_len,))
            if USE_CUDA:
                de_l2_input = de_l2_input.cuda()
            de_l2_output, de_l2_hidden = l2_de(de_l2_input,en_l2_hidden)

            loss = maskMSE(de_l2_output, batch_x, batch_l, recon_fn)
            loss.backward()
            
            en_opt.step()
            l2_de_opt.step()

        print("Finish an epoch")
            

   


