
import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import BertModel, BertTokenizer
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
from src.utils import load_embedding

import logging
logger = logging.getLogger()

class BertTagger(nn.Module):
    def __init__(self, params):
        super(BertTagger, self).__init__()
        self.num_tag = params.num_tag
        self.hidden_dim = params.hidden_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True
        # self.bert = BertModel.from_pretrained("bert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        if params.ckpt != "":
            logger.info("Reloading model from %s" % params.ckpt)
            model_ckpt = torch.load(params.ckpt)
            self.model.load_state_dict(model_ckpt)

        self.linear = nn.Linear(self.hidden_dim, self.num_tag)

    def forward(self, X):
        outputs = self.model(X) # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1] # (bsz, seq_len, hidden_dim)
        
        prediction = self.linear(outputs)

        return prediction


class BiLSTMTagger(nn.Module):
    def __init__(self, params, vocab):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab.n_words, params.emb_dim, padding_idx=0)
        embedding = load_embedding(vocab, params.emb_dim, params.emb_file, params.usechar)
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding))
        
        self.dropout = params.dropout
        self.lstm = nn.LSTM(params.emb_dim, params.lstm_hidden_dim, num_layers=params.n_layer, dropout=params.dropout, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(params.lstm_hidden_dim * 2, params.num_tag)
        self.crf_layer = CRF(params.num_tag)
        
    def forward(self, X, return_hiddens=False):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_tag)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        embeddings = self.embedding(X)
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        lstm_hidden, (_, _) = self.lstm(embeddings)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        if return_hiddens:
            return prediction, lstm_hidden
        else:
            return prediction
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of entity value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(0)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y


class CRF(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via
    backpropagation. 
    """
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats)

    def loss(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and 
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar] 
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))

        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match ', feats.shape, tags.shape)

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        # -ve of l()
        # Average across batch
        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        batch_size = feats.shape[0]

        # Compute feature scores
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # print(feat_score.size())

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for 
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1) # [batch_size, 1, num_tags]
            a = self._log_sum_exp(a.unsqueeze(-1) + transitions + feat, 1) # [batch_size, num_tags]

        return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1) # [batch_size]

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))
        
        v = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i] # [batch_size, num_tags]
            v, idx = (v.unsqueeze(-1) + transitions).max(1) # [batch_size, num_tags], [batch_size, num_tags]
            
            paths.append(idx)
            v = (v + feat) # [batch_size, num_tags]

        
        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)
    
    def _log_sum_exp(self, logits, dim):
        """
        Computes log-sum-exp in a stable way
        """
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()
