
import torch
import torch.nn as nn

from src.conll2002_metrics import *
from src.dataloader import domain2labels, pad_token_label_id

import os
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger()

class BaseTrainer(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_acc = 0

    def train_step(self, X, y):
        self.model.train()

        preds = self.model(X)
        y = y.view(y.size(0)*y.size(1))
        preds = preds.view(preds.size(0)*preds.size(1), preds.size(2))

        self.optimizer.zero_grad()
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_step_for_bilstm(self, X, lengths, y):
        self.model.train()
        preds = self.model(X)
        loss = self.model.crf_loss(preds, lengths, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader, tgt_dm, use_bilstm=False):
        self.model.eval()

        pred_list = []
        y_list = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        if use_bilstm:
            for i, (X, lengths, y) in pbar:
                y_list.extend(y)
                X, lengths = X.cuda(), lengths.cuda()
                preds = self.model(X)
                preds = self.model.crf_decode(preds, lengths)
                pred_list.extend(preds)
        else:
            for i, (X, y) in pbar:
                y_list.extend(y.data.numpy()) # y is a list
                X = X.cuda()
                preds = self.model(X)
                pred_list.extend(preds.data.cpu().numpy())
        
        # concatenation
        pred_list = np.concatenate(pred_list, axis=0)   # (length, num_tag)
        if not use_bilstm:
            pred_list = np.argmax(pred_list, axis=1)
        y_list = np.concatenate(y_list, axis=0)
        
        # calcuate f1 score
        pred_list = list(pred_list)
        y_list = list(y_list)
        lines = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_token = domain2labels[tgt_dm][pred_index]
                gold_token = domain2labels[tgt_dm][gold_index]
                lines.append("w" + " " + pred_token + " " + gold_token)
        results = conll2002_measure(lines)
        f1 = results["fb1"]

        return f1
    
    def train_conll(self, dataloader_train, dataloader_dev, dataloader_test, tgt_dm):
        logger.info("Pretraining on conll2003 NER dataset ...")
        no_improvement_num = 0
        best_f1 = 0
        for e in range(self.params.epoch):
            logger.info("============== epoch %d ==============" % e)
            loss_list = []
        
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for i, (X, y) in pbar:
                X, y = X.cuda(), y.cuda()

                loss = self.train_step(X, y)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

            logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
            f1_dev = self.evaluate(dataloader_dev, tgt_dm)
            logger.info("Evaluate on Dev Set. F1: %.4f." % f1_dev)

            if f1_dev > best_f1:
                logger.info("Found better model!!")
                best_f1 = f1_dev
                no_improvement_num = 0
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, 1))

            # if no_improvement_num >= 1:
            #     break
            if e >= 1:
                break
        
        logger.info("============== Evaluate on Test Set ==============")
        f1_test = self.evaluate(dataloader_test, tgt_dm)
        logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)


    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_finetune_model.pth")
        torch.save({
            "model": self.model,
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)
    

