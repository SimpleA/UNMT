import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, Decoder 
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
def Train(epoch, lr, batch_size, hidden_size, print_every, save_every, postfix, save_dir):
    tr_logfile = open(save_prefix+'_train_logger.csv', 'w')
    print('part loss', file=tr_logfile)

    val_logfile = open(save_prefix+'_valid_logger.csv', 'w')
    print('validation part:', *dataset.validation_list(), sep=' ', file=val_logfile)
    print('validation loss', file=val_logfile)
    val_logfile.close()
    
    stop_cnt = 0
    best_loss = sys.float_info.max
    for nE in range(epochs):
        print('Epoch:',nE,file = tr_logfile)
        en.train()
        de.train()
        dataset.train()
        eStart = time.time()
        print('Start training...')
        for nP, part in enumerate(dataset):
            pLoss = 0
            tStart = time.time()
            for nB, (batch_x, batch_l, _) in enumerate(part):
                seq_len = 58
                batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
                batch_x = batch_x[indices] 
                batch_x = Variable(batch_x).cuda() if USE_CUDA else Variable(batch_x)
                packed_x = pack_padded_sequence(batch_x, batch_l.numpy(), batch_first=True)
                
                en_opt.zero_grad()
                de_opt.zero_grad()
                hidden = en.init_hidden(batch_x.size()[0])
                en_hidden = en(packed_x, hidden)
                
                de_input = Variable(torch.zeros(batch_x.size()[0],seq_len,39))
                if USE_CUDA:
                    de_input = de_input.cuda()
                de_output, de_hidden = de(de_input,en_hidden)

                loss = maskMSE(de_output, batch_x, batch_l, recon_fn)
                loss.backward()
                '''
                clip = 50.0
                torch.nn.utils.clip_grad_norm(en.parameters(), clip)
                torch.nn.utils.clip_grad_norm(de.parameters(), clip)
                '''
                en_opt.step()
                de_opt.step()
            
                pLoss += loss.data[0]

                if (nB+1) % 20 == 0:
                    print('Epoch %3d | Part %s | Batch %3d | loss %6.3f' % (nE+1, dataset.part, nB+1, loss.data[0]/batch_x.size()[0]))
                #del batch_x, packed_x, hidden

            tEnd = time.time()
            print('average loss %6.3f , cost %s'
                  % (pLoss/len(part.dataset), tfmt(tEnd-tStart)))
            print(dataset.part,pLoss/len(part.dataset), sep=',', file=tr_logfile)
            if nP != len(dataset) - 1:
                print('')
        eEnd = time.time()
        print('per epoch cost %s\n' % tfmt(eEnd-eStart))
        val_loss = valid(dataset, en, de, recon_fn,out_prefix, save_prefix,nE)
        if val_loss <= best_loss:
            stop_cnt = 0
            best_loss = val_loss
            torch.save({
                'en':en.state_dict(),
                'de':de.state_dict(),
                'en_opt':en_opt.state_dict(),
                'de_opt':de_opt.state_dict()},
                save_prefix+'_e%s.pkl' % str(nE+1))
        else:
            stop_cnt += 1
        
        if stop_cnt == 4:
            return
