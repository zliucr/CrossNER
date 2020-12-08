
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from src.coach.utils import domain2entitylist, bio_set
from src.dataloader import domain2labels, Vocab, get_vocab, PAD_INDEX
import logging
logger = logging.getLogger()


class Dataset(data.Dataset):
    def __init__(self, X, bin_y, y):
        self.X = X
        self.bin_y = bin_y
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.bin_y[index], self.y[index]
    
    def __len__(self):
        return len(self.X)

def read_ner(datapath, tgt_dm, vocab):
    inputs, labels, bin_labels = [], [], []
    with open(datapath, "r") as fr:
        token_list, label_list, bin_list = [], [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(label_list) == len(bin_list)
                    inputs.append(token_list)
                    labels.append(label_list)
                    bin_labels.append(bin_list)
                
                token_list, label_list, bin_list = [], [], []
                continue
            
            splits = line.split("\t")
            token = splits[0]
            label = splits[1]
            
            token_list.append(vocab.word2index[token])
            label_list.append(domain2labels[tgt_dm].index(label))
            bin_list.append(bio_set.index(label[0]))  # get the B, I or O label

    return inputs, labels, bin_labels

def collate_fn(data):
    X, y_bin, y_final = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    
    return padded_seqs, lengths, y_bin, y_final

def get_dataloader_for_coach(params):
    vocab_src = get_vocab("ner_data/conll2003/vocab.txt")
    vocab_tgt = get_vocab("ner_data/%s/vocab.txt" % params.tgt_dm)

    vocab = Vocab()
    vocab.index_words(vocab_src)
    vocab.index_words(vocab_tgt)

    logger.info("Load training set data ...")
    conll_inputs_train, conll_labels_train, conll_bin_labels_train = read_ner("ner_data/conll2003/train.txt", params.tgt_dm, vocab)
    inputs_train, labels_train, bin_labels_train = read_ner("ner_data/%s/train.txt" % params.tgt_dm, params.tgt_dm, vocab)
    inputs_train = inputs_train * 10 + conll_inputs_train
    labels_train = labels_train * 10 + conll_labels_train
    bin_labels_train = bin_labels_train * 10 + conll_bin_labels_train

    logger.info("Load dev set data ...")
    inputs_dev, labels_dev, bin_labels_dev = read_ner("ner_data/%s/dev.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("Load test set data ...")
    inputs_test, labels_test, bin_labels_test = read_ner("ner_data/%s/test.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, bin_labels_train, labels_train)
    dataset_dev = Dataset(inputs_dev, bin_labels_dev, labels_dev)
    dataset_test = Dataset(inputs_test, bin_labels_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test, vocab
