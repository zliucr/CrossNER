
import torch
import torch.nn as nn
from src.dataloader import domain2labels
from src.coach.utils import domain2entitylist, load_entity_embedding
from src.model import BiLSTMTagger

class EntityPredictor(nn.Module):
    def __init__(self, params):
        super(EntityPredictor, self).__init__()
        self.input_dim = params.lstm_hidden_dim * 2
        self.lstm_enc = nn.LSTM(self.input_dim, params.entity_enc_hidden_dim, num_layers=params.entity_enc_layers, bidirectional=True, batch_first=True)
        self.y_set = domain2labels[params.tgt_dm]
        self.entity_list = domain2entitylist[params.tgt_dm]
        self.entity_embs = load_entity_embedding(params.tgt_dm, params.emb_dim, params.emb_file, params.usechar)
    
    def forward(self, hidden_layers, binary_preditions=None, binary_golds=None, final_golds=None):
        """
        Inputs:
            hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
            binary_predictions: predictions made by our model (bsz, seq_len)
            binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
            final_golds: used only in the training mode (bsz, seq_len)
        Outputs:
            pred_entityname_list: list of predicted entity names
            gold_entityname_list: list of gold entity names  (only return this in the training mode)
        """
        binary_labels = binary_golds if binary_golds is not None else binary_preditions

        feature_list = []
        if final_golds is not None:
            # only in the training mode
            gold_entityname_list = []

        bsz = hidden_layers.size()[0]
        
        ### collect features of entity and their corresponding labels (gold_entityname) in this batch
        for i in range(bsz):
            # dm_id = domains[i]
            # domain_name = domain_set[dm_id]
            # self.entity_list = domain2entitylist[domain_name]  # a list of entity names

            # we can also add domain embeddings after transformer encoder
            hidden_i = hidden_layers[i]    # (seq_len, hidden_dim)

            ## collect range of entity name and hidden layers
            feature_each_sample = []
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
                gold_entityname_each_sample = []
            
            bin_label = binary_labels[i]
            bin_label = torch.LongTensor(bin_label)
            # get indices of B and I
            B_list = bin_label == 1
            I_list = bin_label == 2
            nonzero_B = torch.nonzero(B_list)
            num_entityname = nonzero_B.size()[0]
            
            if num_entityname == 0:
                feature_list.append(feature_each_sample)
                if final_golds is not None:
                    gold_entityname_list.append(gold_entityname_each_sample)
                continue

            for j in range(num_entityname):
                if j == 0 and j < num_entityname-1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_entityname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])

                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_entityname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of entity name is only 1
                        hiddens_based_entityname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    entity_feats, (_, _) = self.lstm_enc(hiddens_based_entityname)   # (1, subseq_len, hidden_dim)
                    entity_feats = torch.sum(entity_feats, dim=1) # (1, hidden_dim)
                    
                    # entity_feats = torch.sum(entity_feats, dim=1)
                    # entity_feats.squeeze(0) ==> (hidden_dim)
                    feature_each_sample.append(entity_feats.squeeze(0))
                    if final_golds is not None:
                        entity_name = self.y_set[final_gold_each_sample[prev_index]].split("-")[1]
                        gold_entityname_each_sample.append(self.entity_list.index(entity_name))
                
                if j == num_entityname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_entityname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of entity name is only 1
                        hiddens_based_entityname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    entity_feats, (_, _) = self.lstm_enc(hiddens_based_entityname)   # (1, subseq_len, hidden_dim)
                    entity_feats = torch.sum(entity_feats, dim=1)  # (1, hidden_dim)

                    # entity_feats = torch.sum(entity_feats, dim=1)
                    feature_each_sample.append(entity_feats.squeeze(0))

                    if final_golds is not None:
                        entity_name = self.y_set[final_gold_each_sample[curr_index]].split("-")[1]
                        gold_entityname_each_sample.append(self.entity_list.index(entity_name))
                        
                else:
                    prev_index = curr_index
            
            feature_each_sample = torch.stack(feature_each_sample)  # (num_entityname, hidden_dim)
            feature_list.append(feature_each_sample)
            if final_golds is not None:
                gold_entityname_each_sample = torch.LongTensor(gold_entityname_each_sample)   # (num_entityname)
                gold_entityname_list.append(gold_entityname_each_sample)

        ### predict entity names
        pred_entityname_list = []
        for i in range(bsz):
            # dm_id = domains[i]
            # domain_name = domain_set[dm_id]
            
            entity_embs = torch.FloatTensor(self.entity_embs).transpose(0,1).cuda()   # (emb_dim, entity_num)

            feature_each_sample = feature_list[i]  # (num_entityname, hidden_dim)  hidden_dim == emb_dim
            if len(feature_each_sample) == 0:
                # only in the evaluation phrase
                pred_entityname_each_sample = None
            else:
                pred_entityname_each_sample = torch.matmul(feature_each_sample, entity_embs) # (num_entityname, entity_num)
            
            pred_entityname_list.append(pred_entityname_each_sample)

        if final_golds is not None:
            # only in the training mode
            return pred_entityname_list, gold_entityname_list
        else:
            return pred_entityname_list
