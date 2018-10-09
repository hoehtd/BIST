import sys
import string
import re
import random
import pickle
import operator
import time
import multiprocessing

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 1
EOS_token = 2
UNK_token = 3

WORD = 1
POS = 4
TAG = 7
HEAD = 6

random.seed()

class LexManager:
	def __init__(self):
		self.word2index = {'<s>': 1, '</s>': 2, '<unk>': 3}
		self.word2count = {'<s>': 0, '</s>': 0, '<unk>': 0}
		self.index2word = ['<UNDEFINED>', '<s>', '</s>', '<unk>']
		self.n_words = 3

	def addSentence(self, sentence):
		for word in sentence:
			self.addWord(word)

	def addConlluTree(self, sentence, extend = True):
		for w in sentence:
			self.addWord(w[WORD], extend)
		self.addWord('<s>')
		self.addWord('</s>')

	def addWord(self, word, extend = True):
		if word not in self.word2index:
			if extend:
				self.word2index[word] = self.n_words
				self.word2count[word] = 1
				self.index2word.append(word)
				self.n_words += 1
			else:
				self.word2count[UNK_token] += 1
		else:
			self.word2count[word] += 1

	def remap(self):
		for i, w in enumerate(self.index2word[1:]):
			self.word2index[w] = i
		self.n_words = len(index2word) - 1


	def remove_rare(self, size = None, minfreq = None):
		if not size or minfreq:
			return
		abandon = set()
		if size and self.n_words > size:
			rl = sorted(self.word2count.items(), key = operator.itemgetter(1), reverse = True)
			abandon = abandon.union([x[0] for x in rl[size:]])
		if minfreq:
			dis = [x for x, c in self.word2count.items() if c < minfreq]
			abandon = abandon.union(dis)

		abandon.difference_update(['<s>', '</s>', '<unk>'])

		for w in abandon:
			del self.index2word[word2index[w]]
			self.word2count['<unk>'] += self.word2count[w]
			del self.word2index[w]
			del self.word2count[w]

		self.remap()


	def word2id(self, word, dropout = False):
		if word in self.word2index:
			if dropout:
				p = .25/(.25 + self.word2count[word])
				keep = random.choices([0, 1], [p, 1-p])
				if sum(keep) > 0:
					return self.word2index[word]
				else:
					return UNK_token
			else:
				return self.word2index[word]
		else:
			return UNK_token


	def save(self, file):
		with open(file, 'wb') as pout:
			pickle.dump([self.word2index, self.word2count, self.index2word, self.n_words], file = pout)

	def load(self, file):
		with open(file, 'rb') as pin:
			self.word2index, self.word2count, self.index2word, self.n_words = pickle.load(pin)


class RNNEncoder(nn.Module):
	def __init__(self, dictsize, n_layers = 2, n_cells = 128, dropout = .3, emb_dim = 300, feat_dim = 50, pretrained_embs = None, pretrained_feat = None):
		super(RNNEncoder, self).__init__()

		self.n_layers = n_layers
		self.n_cells = n_cells
		self.dropout = dropout
		self.emb_dim = emb_dim
		self.feat_dim = feat_dim
		

		if pretrained_embs:
			self.l_embbeding = nn.Embedding.from_pretrained(pretrained_embs, freeze = True)
		else:
			self.l_embbeding = nn.Embedding(dictsize + 1, emb_dim, padding_idx = 0)

		if feat_dim > 0:
			self.l_featemb = nn.Embedding.from_pretrained(pretrained_feat, freeze = True)

		self.inputsize = emb_dim + feat_dim
		self.l_lstm = nn.LSTM(input_size = self.inputsize, hidden_size = n_cells, num_layers = n_layers, dropout = dropout, batch_first = True, bidirectional = True)


	def forward(self, seq, lens, feat = None):
		# sort sentences by length and pad zeros before this
		x = self.l_embbeding(seq)
		# x = F.normalize(x, p=1, dim=-1)
		if self.feat_dim > 0:
			x = torch.cat([x, self.l_featemb(feat)], dim = -1)

		x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)

		x, _ = self.l_lstm(x)
		return x


class GraphBuilder(nn.Module):
	def __init__(self, dim_input, n_hidden = 100):
		super(GraphBuilder, self).__init__()

		self.dim_input = dim_input
		self.n_hidden = n_hidden

		self.l_denseF = nn.Linear(dim_input, n_hidden)
		self.l_denseB = nn.Linear(dim_input, n_hidden)
		self.l_activation = nn.Tanh()
		self.l_outputF = nn.Linear(n_hidden, 1)
		self.l_outputB = nn.Linear(n_hidden, 1)


	def forward(self, feat_seq):
		feat_seq, sizes = nn.utils.rnn.pad_packed_sequence(feat_seq, batch_first = True)
		xf = self.l_denseF(feat_seq)
		xb = self.l_denseB(feat_seq)
		xf = self.l_outputF(xf)
		xb = self.l_outputB(xb)
		xf = self.l_activation(xf)
		xb = self.l_activation(xb)
		return xf, xb, sizes


def get_batch_loss(xf, xb, sizes, y_gold):
	loss = 0
	for i in range(len(sizes)):
		graph = construct_graph(xf[i], xb[i], sizes[i])
		loss += get_graph_loss(graph, y_gold[i])

	return loss/len(sizes)

def get_batch_accuracy(xf, xb, sizes, y_gold):
	accuracy = torch.tensor(0, device = device, dtype = torch.double)
	for i in range(len(sizes)):
		graph = construct_graph(xf[i], xb[i], sizes[i])
		accuracy += get_graph_accuracy(graph, y_gold[i])

	return torch.div(accuracy, len(sizes))

def construct_graph(score_f, score_b, s):
	# scores include <s> and </s>
	# f -> b : head -> dep
	featmat = score_f[:s-1].reshape([1,s-1]).repeat([s-2, 1])
	featmat = featmat + score_b[1:s-1].repeat([1, s-1])
	return featmat


def get_graph_loss(y_pred, y):
	# for individual sentences, y_pred is a n*(n+1) matrix of weights, y is an n*1 array of heads
	return F.cross_entropy(y_pred, y)

def get_graph_accuracy(y_pred, y):
	y_pred = torch.argmax(y_pred, dim = 1)
	return torch.div(sum(y_pred == y).double(), len(y))


def load_trees(file, vocabulary = None, freeze = False):
	bank = []
	with open(file) as fin:
		buf = []
		for line in fin:
			if line.startswith('#'):
				continue
			if len(line) <= 1:
				if buf != []:
					bank.append(buf)
					buf = []
			else:
				temp = line.rstrip('\n').split('\t')
				if float(temp[0]) == int(float(temp[0])):
					buf.append(temp)

	print(f'{len(bank)} sentences in tree bank')

	if vocabulary == None:
		lm = LexManager()
		for sent in bank:
			lm.addConlluTree(sent)
		return bank, lm
	elif freeze == False:
		for sent in bank:
			vocabulary.addConlluTree(sent)
		return bank, vocabulary
	else:
		for sent in bank:
			vocabulary.addConlluTree(sent, False)
		return bank, vocabulary

def convert_sents(sentences, lexman, dropout = False, scramble = True):
	data = []
	labels = []

	ids = [i for i in range(len(sentences))]
	if scramble:
		random.shuffle(ids)
		random.shuffle(ids)
		random.shuffle(ids)

	for i in ids:
		sent = sentences[i]
		data.append([SOS_token] + [lexman.word2id(w[WORD], dropout) for w in sent] + [EOS_token])
		labels.append([int(w[HEAD]) for w in sent])

	return data, labels

def train(bank, lexman, encoder, builder, optimizer, scheduler, savefile):
	random.seed()
	batch_size = 32
	epochs = 30
	n_samples = 300
	steps = 100
	global_step = 0
	
	timept = time.perf_counter()
	logf = open('training_log_4.txt', 'w')
	for e in range(epochs):
		print(f'Epoch {e+1}')

		data, labels = convert_sents(bank, lexman, dropout = True, scramble = False)

		for s in range(steps):
			for bi in range(0, len(bank), batch_size):
				batch = range(bi, min(bi+batch_size, len(bank)))
				# batch = random.sample(range(len(bank)), batch_size)
				b_lens = [(len(data[i]), i) for i in batch]
				b_lens.sort(reverse = True)
				b_data = [torch.tensor(data[i], device = device) for l,i in b_lens]
				b_labels = [torch.tensor(labels[i], device = device) for l,i in b_lens]
				
				b_data = nn.utils.rnn.pad_sequence(b_data, batch_first = True)
				b_lens = torch.tensor([i[0] for i in b_lens], device = device)

				global_step+=1
				x = encoder.forward(b_data, b_lens)
				# print(x)
				xf, xb, sizes = builder.forward(x)
				# print(xf.size())
				# print(xb.size())
				# print(sizes)
				optimizer.zero_grad()
				loss = get_batch_loss(xf, xb, sizes, b_labels)
				loss.backward()
				# scheduler.step()
				optimizer.step()
				if global_step % 100 == 0:
					print(f'Global step {global_step}, loss: {loss}, {time.perf_counter() - timept} seconds')
					logf.write(f'{global_step}\t{loss}\n')
					timept = time.perf_counter()
			
			save_model(savefile.format(e), encoder, builder)
	logf.close()


def test(bank, lexman, encoder, builder):
	error = 0
	data, labels = convert_sents(bank, lexman)
	
	lens = [(len(data[i]), i) for i in range(len(data))]
	lens.sort(reverse = True)
	data = [torch.tensor(data[i], device = device) for l,i in lens]
	labels = [torch.tensor(labels[i], device = device) for l,i in lens]

	data = nn.utils.rnn.pad_sequence(data, batch_first = True)
	lens = torch.tensor([i[0] for i in lens], device = device)

	x = encoder.forward(data, lens)
	xf, xb, sizes = builder.forward(x)
	error = get_batch_accuracy(xf, xb, sizes, labels)
	loss = get_batch_loss(xf, xb, sizes, labels)

	print(f'Testing loss: {loss}, testing accuracy: {error}')


def save_model(pref, encoder, builder):
	torch.save(encoder.state_dict(), pref + '_encoder.params')
	torch.save(builder.state_dict(), pref + '_builder.params')

def load_model(pref, encoder, builder):
	encoder.load_state_dict(torch.load(pref + '_encoder.params'))
	builder.load_state_dict(torch.load(pref + '_builder.params'))




def main():
	if sys.argv[1] == 'train':
		train_flag = True
	elif sys.argv[1] == 'eval':
		train_flag = False
	else:
		print('Must specify "train" or "eval"')
		exit(1)
	
	if train_flag:
		bank, dic = load_trees('UD_English-EWT/en_ewt-ud-train.conllu')
		dic.save('ud_en_ewt.dict')
	else:
		dic = LexManager()
		dic.load('ud_en_ewt.dict')

	encoder = RNNEncoder(dic.n_words, feat_dim = 0, dropout = 0)
	builder = GraphBuilder(256)

	encoder.cuda()
	builder.cuda()

	optimizer = torch.optim.Adam(list(encoder.parameters())+ list(builder.parameters()), lr = 1e-4)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50000, .1)
	
	if train_flag:
		load_model('model_5/epoch_29', encoder, builder)
		train(bank, dic, encoder, builder, optimizer, None, 'model_5/epoch_{}')
	else:
		load_model('model_5/epoch_28', encoder, builder)
		# print(encoder.state_dict())
		# print(builder.state_dict())
		t_bank, ndic = load_trees('UD_English-EWT/en_ewt-ud-dev.conllu')
		test(t_bank, dic, encoder, builder)





if __name__ == '__main__':
	main()

