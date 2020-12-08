
from src.coach.utils import domain2entitylist, bio_set
from src.dataloader import domain2labels
import torch
import torch.nn as nn

import os
from tqdm import tqdm
import numpy as np
import logging
logger = logging.getLogger()

from src.conll2002_metrics import *

class CoachTrainer(object):
    def __init__(self, params, binary_ner_tagger, entityname_predictor):
        self.params = params
        self.y_set = domain2labels[params.tgt_dm]
        self.binary_ner_tagger = binary_ner_tagger
        self.entityname_predictor = entityname_predictor
        self.lr = params.lr
        model_parameters = [
            {"params": self.binary_ner_tagger.parameters()},
            {"params": self.entityname_predictor.parameters()}
        ]
        # Adam optimizer
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_f1 = 0

        self.stop_training_flag = False
    
    def train_step(self, X, lengths, y_bin, y_final):
        self.binary_ner_tagger.train()
        self.entityname_predictor.train()

        bin_preds, lstm_hiddens = self.binary_ner_tagger(X, return_hiddens=True)

        ## optimize binary_ner_tagger
        loss_bin = self.binary_ner_tagger.crf_loss(bin_preds, lengths, y_bin)
        self.optimizer.zero_grad()
        loss_bin.backward(retain_graph=True)
        self.optimizer.step()

        ## optimize entityname_predictor
        pred_entityname_list, gold_entityname_list = self.entityname_predictor(lstm_hiddens, binary_golds=y_bin, final_golds=y_final)
        for pred_entityname_each_sample, gold_entityname_each_sample in zip(pred_entityname_list, gold_entityname_list):
            if pred_entityname_each_sample is not None:
                assert pred_entityname_each_sample.size()[0] == gold_entityname_each_sample.size()[0]
                loss_entityname = self.loss_fn(pred_entityname_each_sample, gold_entityname_each_sample.cuda())
                self.optimizer.zero_grad()
                loss_entityname.backward(retain_graph=True)
                self.optimizer.step()
        
        return loss_bin.item(), loss_entityname.item()
    
    def evaluate(self, dataloader, tgt_dm, use_bilstm):
        self.binary_ner_tagger.eval()
        self.entityname_predictor.eval()

        binary_preds, binary_golds = [], []
        final_preds, final_golds = [], []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (X, lengths, y_bin, y_final) in pbar:
            binary_golds.extend(y_bin)
            final_golds.extend(y_final)

            X, lengths = X.cuda(), lengths.cuda()
            bin_preds_batch, lstm_hiddens = self.binary_ner_tagger(X, return_hiddens=True)
            bin_preds_batch = self.binary_ner_tagger.crf_decode(bin_preds_batch, lengths)
            binary_preds.extend(bin_preds_batch)

            entityname_preds_batch = self.entityname_predictor(lstm_hiddens, binary_preditions=bin_preds_batch, binary_golds=None, final_golds=None)
            
            final_preds_batch = self.combine_binary_and_entityname_preds(bin_preds_batch, entityname_preds_batch, tgt_dm)
            final_preds.extend(final_preds_batch)
        
        # binary predictions
        binary_preds = np.concatenate(binary_preds, axis=0)
        binary_preds = list(binary_preds)
        binary_golds = np.concatenate(binary_golds, axis=0)
        binary_golds = list(binary_golds)

        # final predictions
        final_preds = np.concatenate(final_preds, axis=0)
        final_preds = list(final_preds)
        final_golds = np.concatenate(final_golds, axis=0)
        final_golds = list(final_golds)

        bin_lines, final_lines = [], []
        for bin_pred, bin_gold, final_pred, final_gold in zip(binary_preds, binary_golds, final_preds, final_golds):
            bin_entity_pred = bio_set[bin_pred]
            bin_entity_gold = bio_set[bin_gold]
            
            final_entity_pred = self.y_set[final_pred]
            final_entity_gold = self.y_set[final_gold]
            
            bin_lines.append("w" + " " + bin_entity_pred + " " + bin_entity_gold)
            final_lines.append("w" + " " + final_entity_pred + " " + final_entity_gold)
            
        bin_result = conll2002_measure(bin_lines)
        bin_f1 = bin_result["fb1"]
        
        final_result = conll2002_measure(final_lines)
        final_f1 = final_result["fb1"]
        
        return final_f1
        
    def combine_binary_and_entityname_preds(self, binary_preds_batch, entityname_preds_batch, tgt_dm):
        """
        Input:
            binary_preds: (bsz, seq_len)
            entityname_preds: (bsz, num_entityname, entity_num)
        Output:
            final_preds: (bsz, seq_len)
        """
        final_preds = []
        for i in range(len(binary_preds_batch)):
            # dm_id = dm_id_batch[i]
            binary_preds = binary_preds_batch[i]
            entityname_preds = entityname_preds_batch[i]
            # entity_list_based_dm = domain2entity[domain_set[dm_id]]
            entity_list_based_dm = domain2entitylist[tgt_dm]
            
            i = -1
            final_preds_each = []
            for bin_pred in binary_preds:
                # values of bin_pred are 0 (O), or 1(B) or 2(I)
                if bin_pred.item() == 0:
                    final_preds_each.append(0)
                elif bin_pred.item() == 1:
                    i += 1
                    pred_entity_id = torch.argmax(entityname_preds[i])
                    entityname = "B-" + entity_list_based_dm[pred_entity_id]
                    final_preds_each.append(self.y_set.index(entityname))
                elif bin_pred.item() == 2:
                    if i == -1:
                        final_preds_each.append(0)
                    else:
                        pred_entity_id = torch.argmax(entityname_preds[i])
                        entityname = "I-" + entity_list_based_dm[pred_entity_id]
                        if entityname not in self.y_set:
                            final_preds_each.append(0)
                        else:
                            final_preds_each.append(self.y_set.index(entityname))
            
            assert len(final_preds_each) == len(binary_preds)

            final_preds.append(final_preds_each)

        return final_preds
    
    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
            "binary_ner_tagger": self.binary_ner_tagger,
            "entityname_predictor": self.entityname_predictor
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)
