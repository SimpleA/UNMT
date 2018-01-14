from load import Vocab
from utils import USE_CUDA
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model import Encoder, Attn, AttnDecoder
from load import LanguageDataset,MAX_LENGTH, VOCAB_SIZE
from load import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
import pickle
import numpy as np
import tqdm
import pdb

class TestDataset(Dataset):
	def __init__(self, data_path, language, verbose=True):
		self.vocab = Vocab(data_path, language, True)
		self.emb = {}
		try:
			saved = open('data/emb_{}.dms'.format(language), 'rb')
			if verbose:
				print("Found existing embedding, loading...")
			self.emb = pickle.load(saved)
		except:
			if verbose:
				print("Saved embedding not found, create one...")
			with open('data/cross_{}'.format(language), 'r', encoding='utf-8') as f:
				f.readline()
				for line in f:
					info = line.strip().split()
					self.emb[info[0]] = torch.FloatTensor([float(num) for num in info[1:]])
			of = open('data/emb_{}.dms'.format(language), 'wb')
			pickle.dump(self.emb, of)
		# read test data
		self.sentence = []
		self.sent_id = []
		with open('data/test_{}.proc'.format(language), 'r', encoding='utf-8') as tp:
			for idx, line in enumerate(tp):
				line = line.strip()
				fields = line.split(',')
				sent = ' '.join(fields[1:])
				if len(sent.split(' ')) < 50:
					self.sent_id.append(fields[0])
					self.sentence.append(sent)

	def sen2index(self, sen):
		sen_emb = torch.zeros(50, 300)
		for ch in '.!()':
			if ch in sen:
				sen = sen.replace(ch, '') 
		sen_index = [PAD_TOKEN for _ in range(MAX_LENGTH)]
		length = 0
		for idx, word in enumerate(sen.split()):
			if word in self.vocab.word2index.keys():
				sen_index[idx] = self.vocab.word2index[word]
			else:
				sen_index[idx] = UNK_TOKEN
			if word in self.emb.keys():
				sen_emb[idx] = self.emb[word]
			else:
				# Unkown word
				sen_emb[idx] = self.emb[self.vocab.index2word[UNK_TOKEN]]
			length += 1
		return torch.LongTensor(sen_index), length, sen_emb

	def get_embed(self, idx):
		word = self.vocab.index2word[idx]
		if word in self.emb.keys():
			embed = self.emb[word]
		else:
			embed = self.emb[self.vocab.index2word[UNK_TOKEN]]
		return embed
	
	def __len__(self):
		return len(self.sentence)

	def __getitem__(self, idx):
		return self.sen2index(self.sentence[idx])

def Test(l1, l2, batch_size, hidden_size, vocab_size, model_path):
		print('Restoring saved model...')
		# Contruct graph
		checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
		input_size = 300
		output_size = vocab_size
		en = Encoder(input_size, hidden_size)
		l1_de = AttnDecoder('general', input_size, hidden_size, output_size)
		l2_de = AttnDecoder('general', input_size, hidden_size, output_size)
		# Load model
		en.load_state_dict(checkpoint['en'])
		l1_de.load_state_dict(checkpoint['l1_de'])
		l2_de.load_state_dict(checkpoint['l2_de'])

		print('Loading test dataset')
		l1_dataset = TestDataset('data/test_{}'.format(l1), l1)
		l2_dataset = TestDataset('data/test_{}'.format(l2), l2)
		l1_loader = DataLoader(l1_dataset, batch_size=batch_size, shuffle=False)
		l2_loader = DataLoader(l2_dataset, batch_size=batch_size, shuffle=False)

		# Translation from l1 to l2
		for batch_index, batch_l, batch_sentence in l1_loader:
			pdb.set_trace()
			bsz = len(batch_l)
			if USE_CUDA:
				batch_index = batch_index.cuda()
				batch_l = batch_l.cuda()
				batch_sentence = batch_sentence.cuda()

			batch_sentence = Variable(batch_sentence)
			batch_l, indices = torch.sort(batch_l, dim=0, descending=True)
			batch_sentence = batch_sentence[indices]
			batch_index = Variable(batch_index[indices])
			packed_sentence = pack_padded_sequence(batch_sentence.float(), batch_l.cpu().numpy(), batch_first=True)

			l2_de.eval()

			# Encode l1
			l2_hidden = en.init_hidden(batch_sentence.size(0))
			packed_output, en_l2_hidden = en(packed_sentence, l2_hidden)
			output,_ = pad_packed_sequence(packed_output,batch_first = True)

			# Init. l2 decoder
			de_l2_input = Variable(torch.zeros(bsz,1,input_size)) 
			de_l2_context = Variable(torch.zeros(bsz,1,2*hidden_size))
			if USE_CUDA:
				de_l2_input = de_l2_input.cuda()
				de_l2_context = de_l2_context.cuda()
			de_l2_hidden_0 = (en_l2_hidden[0]+en_l2_hidden[1]).unsqueeze(0)
			de_l2_hidden_1 = (en_l2_hidden[2]+en_l2_hidden[3]).unsqueeze(0)
			de_l2_hidden = torch.cat((de_l2_hidden_0,de_l2_hidden_1),0)

			de_l2_output = Variable(torch.zeros(bsz, MAX_LENGTH, 1, input_size)).cuda() if USE_CUDA else Variable(torch.zeros(bsz, MAX_LENGTH, 1, input_size))

			# Decode l2
			txt_output = []
			for di in range(output.size(1)):
				if di == 0:
					uni_output, de_l2_context, de_l2_hidden, de_l2_attention = l2_de(de_l2_input, de_l2_context, de_l2_hidden, output)
				else:
					uni_output, de_l2_context, de_l2_hidden, de_l2_attention = l2_de(de_l2_output[:,di-1,:,:], de_l2_context, de_l2_hidden, output)
				_,choose = torch.max(uni_output,1)
				print(choose)
				pdb.set_trace()
				de_l2_input = [l2_dataset.get_embed(index.data[0]).tolist() for index in choose]
				de_l2_input = torch.FloatTensor(de_l2_input).unsqueeze(1)
				de_l2_output[:,di,:,:] = de_l2_input
				txt_output.append([l2_dataset.vocab.index2word[index.data[0]] for index in choose])

			txt_output = np.asarray(txt_output).transpose()
			with open('l1l2.txt', 'a', encoding='utf-8') as l1_out:
				for idx in range(txt_output.shape[0]):
					l1_out.write(' '.join(txt_output[idx].tolist()) + '.\n')
		
