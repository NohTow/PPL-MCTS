
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from random import randint
import os
os.environ['TRANSFORMERS_CACHE'] = 'placeholder'
from transformers import BertTokenizer, BertModel, GPT2TokenizerFast, AdamW, get_linear_schedule_with_warmup
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
torch.multiprocessing.freeze_support()

class TextDataset(torch.utils.data.DataLoader):
    """ Textual dataset from tsv """
    def __init__(self, set_name):
        self.file = pd.read_csv(path + "/" + set_name + ".tsv", delimiter='\t', encoding="utf-8")
    def __len__(self):
        return len(self.file)
    def __getitem__(self, idx):
        item = self.file.iloc[idx]
        return "<|startoftext|> " + str(item["text"]) + "<|endoftext|>", item["label"]
    
    
#-------------- Model definition ---------------#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.fc_txt1 = nn.Linear(768, 512)
        self.fc_txt2 = nn.Linear(512, 256)
        self.fc_classif = nn.Linear(256, 6)



    def forward(self, texts):
        tokenizer_res = tokenizer.batch_encode_plus(texts, truncation=True, max_length=512, padding='longest')
        tokens_tensor = torch.cuda.LongTensor(tokenizer_res['input_ids']) 
        attention_tensor = torch.cuda.LongTensor(tokenizer_res['attention_mask'])
        output = self.bert(tokens_tensor, attention_mask=attention_tensor)
        text = F.normalize(torch.div(torch.sum(output[2][-1], axis=1),torch.unsqueeze(torch.sum(attention_tensor, axis=1),1)))
        text = F.relu(self.fc_txt1(text))
        text = F.relu(self.fc_txt2(text))
        text = self.fc_classif(text)
        return nn.Softmax(dim = 1)(text).cpu()

#-------------- Training ---------------#
import pickle
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default=None,
    type=str,
    required=True,
    help="Name of the dataset to generate",
)
args = parser.parse_args()
batch_size = 5
nb_epoch = 75

path = "placeholder" + args.dataset
# Prepare the data
data = pd.read_csv("datasets/emotion/full/train_2.tsv", sep='\t', engine='python', encoding="utf8")
labels_count = np.zeros(data["label"].nunique())
for index, row in data.iterrows():
    labels_count[int(row["label"])] += 1
weight = np.ones(data["label"].nunique())
weight = weight / labels_count
samples_weight = np.array([weight[label] for label in data["label"]])
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
print("loading training dataset")
dataset_train = TextDataset("train_2") # create the dataset
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=8)
print("training dataset loaded")

print("loading validation dataset")
dataset_validate = TextDataset("validation_2") # create the dataset
valid_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=8)
print("validation dataset loaded")

# Create an instance of our network
net = Net()
net.cuda()
# Define the optimization criterion
criterion = nn.CrossEntropyLoss().cuda()
optimizer = AdamW(net.parameters(), lr=2e-5, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=nb_epoch * len(train_loader)
)

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Train
best = 0.0
for epoch in range(nb_epoch):
    pbar=tqdm(total=len(dataset_train.file))
    running_loss = 0.0
    for batch_idx, (inputs_txt_train, labels_train) in enumerate(train_loader):
        outputs = net(inputs_txt_train)
        loss = criterion(outputs, labels_train.long())
        running_loss = running_loss + loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        pbar.update(batch_size)
            
    correct = 0
    total = 0
    pos = 0
    pred_one = 0
    res_preds = []
    res_labels = []
    print("Epoch : {}, loss : {}".format(epoch, running_loss))
    pbar_val = tqdm(total=len(dataset_validate.file))
    with torch.no_grad():
        for batch_idx, (inputs_txt_valid, labels_valid) in enumerate(valid_loader):
            outputs = net(inputs_txt_valid)
            predicted = torch.argmax(outputs.data, 1)
            res_preds.extend(predicted.numpy())
            res_labels.extend(labels_valid.numpy().astype(int))
            total += labels_valid.size(0)
            pred_one += (predicted==1).sum().item()
            correct += (predicted == labels_valid).sum().item()
            pbar_val.update(batch_size)
            
    if(100 * correct / total > best):
        best = 100 * correct / total
        torch.save(net.state_dict(), path + "/models/validation_BEST_bert_tuned_oracle" + timestr + ".pth")
    print('Epoch %d Accuracy of the network on the %d valid packages : %d %%, best : %d %%' % (epoch, total,
100 * correct / total, best))
    print(confusion_matrix(res_labels, res_preds))
    
        


torch.save(net.state_dict(), path + "/models/validation_final_bert_tuned_oracle" + timestr + ".pth")


