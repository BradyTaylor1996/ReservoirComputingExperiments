import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from dsn import *
from trainer import *
from extract import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = "./sleep-edfx/cassette_npz/train/"
train_records = [train_dir+f for f in listdir(train_dir) if(f.find('.npz') != -1)]
test_dir = "./sleep-edfx/cassette_npz/test/"
test_records = [test_dir+f for f in listdir(test_dir) if(f.find('.npz') != -1)]
val_dir = "./sleep-edfx/cassette_npz/val/"
val_records = [val_dir+f for f in listdir(val_dir) if(f.find('.npz') != -1)]

records = train_records + test_records + val_records
records = np.array(records)
data_size = len(records)
permute = np.random.permutation(data_size)

# Data is too large to run locally on laptop
# train_records = records[permute[:int(data_size*0.6)]]
# val_records = records[permute[int(data_size*0.6): int(data_size*0.8)]]
# test_records = records[permute[int(data_size*0.8):]]
train_records = records[permute[0:2]]
val_records = records[permute[2:3]]
test_records = records[permute[3:4]]

# We get a random collection of files (usually) for training, validation, and testing
# Then the batches are shuffled (which is why we store sequences)

model_1 = Extractor()
model_1.apply(weights_init)
model_1.to(device)

train_x = []
train_y = []
for sf in train_records:
    with np.load(sf) as f:
        # print(sf)
        x = f['x']
        y = f['y']
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        batch_x, batch_y = process(x, y)
        train_x.append(torch.tensor(batch_x))
        train_y.append(torch.tensor(batch_y))
train_x = torch.cat(train_x)
train_y = torch.cat(train_y)
#print(train_x.shape, train_y.shape)

train_f = []
with torch.no_grad():
    for seq in train_x:
    	outputs = model_1(seq.to(device))
    	train_f.append(outputs.cpu().detach())
train_feats = torch.stack(train_f)
# Output is >100 sequences of 25 windows, each output 64 x 296 (num kernels x filtered length)
# At the moment, I don't think it matters in what order elements of each sequence are added (?)

train_set = TensorDataset(train_feats, train_y)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

test_x = []
test_y = []
for sf in test_records:
    with np.load(sf) as f:
        # print(sf)
        x = f['x']
        y = f['y']
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        batch_x, batch_y = process(x, y)
        test_x.append(torch.tensor(batch_x))
        test_y.append(torch.tensor(batch_y))
test_x = torch.cat(test_x)
test_y = torch.cat(test_y)
#print(test_x.shape, test_y.shape)

test_f = []
with torch.no_grad():
    for seq in test_x:
    	outputs = model_1(seq.to(device))
    	test_f.append(outputs.cpu().detach())
test_feats = torch.stack(test_f)

test_set = TensorDataset(test_feats, test_y)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)


val_x = []
val_y = []
for sf in val_records:
    with np.load(sf) as f:
        # print(sf)
        x = f['x']
        y = f['y']
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        batch_x, batch_y = process(x, y)
        val_x.append(torch.tensor(batch_x))
        val_y.append(torch.tensor(batch_y))
val_x = torch.cat(val_x)
val_y = torch.cat(val_y)
#print(val_x.shape, val_y.shape)

val_f = []
with torch.no_grad():
    for seq in val_x:
    	outputs = model_1(seq.to(device))
    	val_f.append(outputs.cpu().detach())
val_feats = torch.stack(val_f)

val_set = TensorDataset(val_feats, val_y)
val_loader = DataLoader(val_set, batch_size=10, shuffle=False)

# print(train_feats.shape, test_feats.shape, val_feats.shape)

# I think there does need to be some sort of longer timescale processing...

lr = 1e-4

model = DeepSleepNet()
model.apply(weights_init)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

EEG_trainer = trainer(model, optimizer, criterion, './checkpoint/cnn_base.pt', 100)
EEG_trainer.train(train_loader, test_loader)


def cal_sens_spec(y_true, y_pred, type_label):
    type_y_true = np.where(y_true==type_label, 1, 0)
    type_y_pred = np.where(y_pred==type_label, 1, 0)
    tn, fp, fn, tp = confusion_matrix(type_y_true, type_y_pred).ravel()
    return tp / (tp + fp), tp / (tp+fn), tn / (tn + fp), accuracy_score(type_y_true, type_y_pred), \
            tp / (tp + (fp+fn)/2)


class_names = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'REM'] # If you want Stage 4, change load_data.py

y_true = []
y_pred = []
model.load_state_dict(torch.load('./checkpoint/cnn_base.pt', map_location=device))

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        tag_scores = model(inputs.to(device))
        tag_pred = torch.argmax(tag_scores, dim=1)
        y_pred.append(np.asarray(tag_pred.cpu().detach()).flatten())
        #         targets = targets.to(device)
        y_true.append(np.asarray(targets.cpu().detach()).flatten())

y_pred, y_true = np.concatenate(y_pred), np.concatenate(y_true)
global_acc = accuracy_score(y_true, y_pred)
print('test_acc: ', global_acc)

res = [[], [], [], [], []]
for i in range(5):
    ppv, sens, spec, acc, f1 = cal_sens_spec(y_true, y_pred, i)
    print('%s PPV: %.3f, Sensitivity: %.3f, Specificity: %.3f, Accuracy: %.3f, F1: %.3f' \
          % (class_names[i], ppv, sens, spec, acc, f1))
    res[i].append(ppv)
    res[i].append(sens)
    res[i].append(spec)
    res[i].append(f1)

kappa = cohen_kappa_score(y_true, y_pred)
res = np.asarray(res)
res = np.mean(res, axis=0)
print('Overall PPV: %.3f, Sensitivity: %.3f, Specificity: %.3f, Accuracy: %.3f, F1: %.3f, Kappa: %.3f' \
          % (res[0], res[1], res[2], global_acc, res[3], kappa))

