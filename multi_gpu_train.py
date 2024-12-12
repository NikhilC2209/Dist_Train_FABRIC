import pandas as pd
import numpy as np

imdb_df = pd.read_csv('IMDB_Dataset2.csv', index_col = None, on_bad_lines='skip')

print(imdb_df.head(3))

## Label sentiments as 0 or 1
## 1 --> positive
## 0 --> negative

label = []

for index, row in imdb_df.iterrows():
    if(row['sentiment'] == 'positive'): label.append(1)
    else: label.append(0)


imdb_df['label'] = label

imdb_df = imdb_df.drop(['sentiment'], axis = 1)

print("\n\n\nData after labelling\n\n\n")

print(imdb_df.head(3))

## Import Roberta model

from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

## For local models

#PRETRAINED_MODEL_NAME = 'roberta-large'
#PRETRAINED_MODEL_PATH = '../models/' + PRETRAINED_MODEL_NAME

roberta_model = RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-large')
roberta_tok = RobertaTokenizer.from_pretrained('FacebookAI/roberta-large')

## Creating a dataset

import torch
from sklearn.model_selection import train_test_split

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(review, 
                                             add_special_tokens = True,
                                             max_length = self.max_len, 
                                             truncation = True,
                                             return_tensors = 'pt',
                                             return_token_type_ids = False,
                                             return_attention_mask = True,
                                             padding = 'max_length')
        
        return{
            'review_text': review,
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'labels' : torch.tensor(label, dtype=torch.long)            
        }

df_train, df_val = train_test_split(imdb_df.iloc[:1000,:], test_size = 0.3, random_state = 2021)
print(df_train.shape, df_val.shape)

## Create Dataloader

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CreateDataset(reviews = df.review.to_numpy(),
                       labels = df.label.to_numpy(),
                       tokenizer = tokenizer,
                       max_len = max_len
                      )
    
    return torch.utils.data.DataLoader(ds, 
                                       batch_size = batch_size, 
                                       num_workers = 4)

MAX_LEN = 512
BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, roberta_tok, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, roberta_tok, MAX_LEN, BATCH_SIZE)

check_data = next(iter(train_data_loader))
print(check_data.keys())
#print(check_data)

## Multi-GPU architecture

class MultiGPUClassifier(torch.nn.Module):
    def __init__(self, roberta_model):
        super(MultiGPUClassifier, self).__init__()
        self.embedding = roberta_model.roberta.embeddings.to('cuda:0')
        self.encoder = roberta_model.roberta.encoder.to('cuda:1')
        self.classifier = roberta_model.classifier.to('cuda:1')
        
    def forward(self, input_ids, token_type_ids = None, attention_mask = None, labels = None):
        emb_out = self.embedding(input_ids.to('cuda:0'))
        enc_out = self.encoder(emb_out.to('cuda:1'))
        classifier_out = self.classifier(enc_out[0])
        return classifier_out      

## Degine Training PARAMS

multi_gpu_roberta = MultiGPUClassifier(roberta_model)

from transformers import get_linear_schedule_with_warmup, AdamW

EPOCHS = 10 
LR = 1e-5

optimizer = AdamW(multi_gpu_roberta.parameters(), lr = LR)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                           num_warmup_steps = 0, 
                                           num_training_steps = total_steps)

loss_fn = torch.nn.CrossEntropyLoss().to('cuda:1')

def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        
        #print("hereeeeeeeeeeeeeeeeeeeeeeee")
        
        input_ids = d['input_ids']
        attention_mask = d['attention_mask']
        reshaped_attention_mask = attention_mask.reshape(d['attention_mask'].shape[0], 1, 1, d['attention_mask'].shape[1])
        targets = d['labels']
        
        #print("hereeeeeeeeeeeeeeeee2222222")

        outputs= model(input_ids = input_ids, attention_mask = reshaped_attention_mask)
        _, preds = torch.max(outputs, dim = 1)
        loss = loss_fn(outputs, targets.to('cuda:1'))
        
        correct_predictions += torch.sum(preds == targets.to('cuda:1'))
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
                scheduler.step()
        optimizer.zero_grad()
       
       # print("hereeeeeeeeeee33333333333")

        #print(correct_predictions.double() / n_examples, np.mean(losses))
 
    return correct_predictions.double() / n_examples, np.mean(losses)

## EVAL PREDICTIONS

def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids']
            attention_mask = d['attention_mask']
            reshaped_attention_mask = attention_mask.reshape(d['attention_mask'].shape[0], 1, 1, d['attention_mask'].shape[1])
            targets = d['labels']
            
            outputs = model(input_ids = input_ids, attention_mask = reshaped_attention_mask)
            _, preds = torch.max(outputs, dim = 1)
            
            loss = loss_fn(outputs, targets.to('cuda:1'))
            
            correct_predictions += torch.sum(preds == targets.to('cuda:1'))
            losses.append(loss.item())
            
        return correct_predictions.double() / n_examples, np.mean(losses)

from collections import defaultdict

history = defaultdict(list)
best_accuracy = 0

## TRAINING LOOP

import time

start = time.time()

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    train_acc, train_loss = train_model(multi_gpu_roberta, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
    print(f'Train Loss: {train_loss} ; Train Accuracy: {train_acc}')
    
    val_acc, val_loss = eval_model(multi_gpu_roberta, val_data_loader, loss_fn, len(df_val))
    print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')
    
    print()
    
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    
    if val_acc > best_accuracy:
        torch.save(multi_gpu_roberta.state_dict(), 'multi_gpu_roberta_best_model_state.bin')
        best_acc = val_acc

end = time.time()

print("Training data for Batch size 32 over multiple GPU's")
print(end-start)