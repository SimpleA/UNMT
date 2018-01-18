from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import pdb

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
VOCAB_SIZE = 49000
MAX_LENGTH = 50


class Vocab:
    def __init__(self, data_path, language, verbose):
        self.verbose = verbose
        if verbose:
            print("Building Vocab...")
        self.path = data_path
        self.language = language
        self.word2index = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.index2word = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word2count = {'<PAD>': 1, '<BOS>': 1, '<EOS>': 1, '<UNK>': 1}
        self.num_words = 4
        self.sentence = []
        self.build()

    def build(self):
        try:
            sf = open('data/vocab_{}.pkl'.format(self.language), 'rb')
            if self.verbose:
                print("Found existing vocab and sentence, loading...")
            saved = pickle.load(sf)
            self.word2index = saved['word2index']
            self.index2word = saved['index2word']
            self.sentence = saved['sentence']
            self.word2count = saved['word2count']
            sf.close()
        except:
            if self.verbose:
                print("Saved file not found! Creating...")
            with open(self.path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if len(line.strip().split()) <= MAX_LENGTH:
                        self.sentence.append(line.strip())
                    for ch in '.!()':
                        if ch in line:
                            line = line.replace(ch, '')
                    for w in line.strip().split():
                        if w not in self.word2count.keys():
                            self.word2count[w] = 1
                        else:
                            self.word2count[w] += 1

            sorted_word = [w for (w, c) in sorted(
                self.word2count.items(), key=lambda x: x[1], reverse=True)]
            for w in sorted_word[:VOCAB_SIZE-1]:
                if w in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                    continue
                self.word2index[w] = self.num_words
                self.index2word[self.num_words] = w
                self.num_words += 1
            output_dict = open('data/vocab_{}.pkl'.format(self.language), 'wb')
            pickle.dump({'word2count': self.word2count, 'word2index': self.word2index,
                         'index2word': self.index2word, 'sentence': self.sentence}, output_dict)
            output_dict.close()


class LanguageDataset(Dataset):
    def __init__(self, data_path, language, verbose):
        self.vocab = Vocab(data_path, language, verbose)
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
    def sen2index(self, sen):
        sen_emb = torch.zeros((MAX_LENGTH, 300))
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
                # Do not use subword encoding
                # Search for subword. If match, replace UNK with the subword embedding
                # for char_idx in reversed(range(1, len(word) + 1)):
                #     subword = word[:char_idx] + '@@'
                #     if subword in self.emb.keys():
                #         sen_emb[idx] = self.emb[subword]
                #         break
            length += 1
        # No need to handle EOS since EOS is already appended to the end of every sentence
        # sen_index[len(sen.split())] = EOS_TOKEN
        return torch.LongTensor(sen_index), length, sen_emb

    def get_embed(self, idx):
        word = self.vocab.index2word[idx]
        if word in self.emb.keys():
            embed = self.emb[word]
        else:
            embed = self.emb[self.vocab.index2word[UNK_TOKEN]]
            # DO not use subword encoding
            # for char_idx in reversed(range(1, len(word) + 1)):
            #     subword = word[:char_idx] + '@@'
            #     if subword in self.emb.keys():
            #         embed = self.emb[subword]
            #         break

        return embed

    def __len__(self):
        return len(self.vocab.sentence)

    def __getitem__(self, idx):
        return self.sen2index(self.vocab.sentence[idx])
