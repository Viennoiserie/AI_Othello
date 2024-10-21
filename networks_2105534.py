import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from tqdm import tqdm
import pandas as pd
import numpy as np

import os
import sys
import h5py
import json
import copy
import time

from datetime import datetime


#----------------------------- FIN DES IMPORTS -------------------------------

BOARD_SIZE=8

def loss_fnc(predictions, targets):

    return nn.CrossEntropyLoss()(input=predictions,target=targets)



class MLP(nn.Module):

    def __init__(self, conf):

        super(MLP, self).__init__()  
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)

        self.dropout = nn.Dropout(p = 0.1)
        

    def forward(self, seq):

        seq = np.squeeze(seq)

        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim = 1)

        else:
            seq = torch.flatten(seq, start_dim = 0)

        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)

        return F.softmax(outp, dim = -1)
    

    def train_all(self, train, dev, num_epoch, device, optimizer):

        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")

        best_dev = 0.0
        dev_epoch = 0
        notchange = 0 # Used to manage earlystopping

        train_acc_list = []
        dev_acc_list = []

        torch.autograd.set_detect_anomaly(True)

        init_time = time.time()

        for epoch in range(1, num_epoch+1):

            start_time = time.time()

            loss = 0.0
            nb_batch =  0
            loss_batch = 0

            for batch, labels, _ in tqdm(train):

                outputs =self(batch.float().to(device))

                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                nb_batch += 1
                loss_batch += loss.item()

            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' + str(loss_batch/nb_batch))
            last_training = time.time() - start_time

            self.eval()
            
            train_clas_rep = self.evaluate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evaluate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch

            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()

        _clas_rep = self.evaluate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch
    
    
    def evaluate(self, test_loader, device):
        
        all_predicts = []
        all_targets =  []
        
        for data, target, _ in tqdm(test_loader):

            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()

            for i in range(len(predicted)):

                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division = 1,
                                         digits = 4,
                                         output_dict = True)

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division = 1,
                                         digits = 4,
                                         output_dict = True)
        
        return perf_rep
    
    

class LSTMs(nn.Module):

    def __init__(self, conf):

        super(LSTMs, self).__init__()
        
        self.board_size  = conf["board_size"]
        self.path_save = conf["path_save"]+"_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        self.hidden2output = nn.Linear(self.hidden_dim*2, self.board_size*self.board_size)
        # self.hidden2output = nn.Linear(self.hidden_dim, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p = 0.1)


    def forward(self, seq):
        
        seq = np.squeeze(seq)

        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim = 2)

        else:
            seq=torch.flatten(seq, start_dim = 1)

        lstm_out, (hn, cn) = self.lstm(seq)

        outp = self.hidden2output(torch.cat((hn,cn),-1))
        # outp = self.hidden2output(lstm_out)
        outp = F.softmax(outp, dim = 1).squeeze()

        return outp

    
    def train_all(self, train, dev, num_epoch, device, optimizer):

        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")

        best_dev = 0.0
        dev_epoch = 0
        notchange = 0

        train_acc_list=[]
        dev_acc_list=[]

        torch.autograd.set_detect_anomaly(True)

        init_time = time.time()

        for epoch in range(1, num_epoch+1):

            start_time=time.time()

            loss = 0.0
            nb_batch =  0
            loss_batch = 0

            for batch, labels, _ in tqdm(train):

                outputs = self(batch.float().to(device))

                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                nb_batch += 1
                loss_batch += loss.item()

            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+ str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evaluate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evaluate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch

            else:
                notchange += 1

                if notchange > self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()

        _clas_rep = self.evaluate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    
    def evaluate(self, test_loader, device):
        
        all_predicts = []
        all_targets =  []
        
        for data, target_array, lengths in tqdm(test_loader):

            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()

#             import pdb
#             pdb.set_trace()

            for i in range(len(predicted)):

                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division = 1,
                                         digits = 4,
                                         output_dict = True)

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division = 1,
                                         digits = 4,
                                         output_dict = True)
        
        return perf_rep

class ConvNeuralNet(torch.nn.Module):
    
    def __init__(self, conf):

        super(ConvNeuralNet, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"]+"_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        self.conv_layer1 = torch.nn.Conv2d(in_channels=1, out_channels = 32, kernel_size = (2, 2))
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 1)
        
        self.conv_layer2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (2, 2))
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 1)
        
        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(1024,  self.board_size * self.board_size)

        self.fc2 = torch.nn.Linear(self.board_size * self.board_size,  self.board_size * self.board_size)

    
    def forward(self, x):

        x = np.reshape(x, (x.shape[0], 1, x.shape[-2], x.shape[-1]))

        out = self.conv_layer1(x)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.relu1(out)
        out = self.fc1(out)
        #return(F.softmax(out, dim=-1))
        return(out)


    def train_all(self, train, dev, num_epoch, device, optimizer):

        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")

        best_dev = 0.0
        dev_epoch = 0
        notchange = 0

        train_acc_list=[]
        dev_acc_list=[]

        torch.autograd.set_detect_anomaly(True)

        init_time = time.time()

        for epoch in range(1, num_epoch+1):

            start_time=time.time()

            loss = 0.0
            nb_batch =  0
            loss_batch = 0

            for batch, labels, _ in tqdm(train):

                outputs = self(batch.float().to(device))

                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                nb_batch += 1
                loss_batch += loss.item()

            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+ str(loss_batch/nb_batch))
            last_training = time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evaluate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evaluate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch

            else:
                notchange += 1

                if notchange > self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()

        _clas_rep = self.evaluate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        
        return best_epoch
    
    
    def evaluate(self, test_loader, device):
        
        all_predicts = []
        all_targets =  []
        
        for data, target_array, lengths in tqdm(test_loader):

            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()

#             import pdb
#             pdb.set_trace()

            for i in range(len(predicted)):

                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division = 1,
                                         digits = 4,
                                         output_dict = True)

        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division = 1,
                                         digits = 4,
                                         output_dict = True)
        
        return perf_rep


            

