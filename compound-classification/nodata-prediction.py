import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, T5EncoderModel, get_linear_schedule_with_warmup
import datasets
from datasets import load_dataset, load_metric
import sentencepiece
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
import pickle
import time
import math
from sklearn.preprocessing import MinMaxScaler
from datasets.utils.logging import disable_progress_bar
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
disable_progress_bar()

class CFG():
    data_path='../../data_for_classification.csv'
    pretrained_model_name_or_path='sagawa/ZINC-t5'
    debug = True
    epochs = 5
    lr = 2e-5
    batch_size = 15
    max_len = 128
    weight_decay = 0.01
    seed = 42
    num_workers = 4
    fc_dropout = 0.1
    eps = 1e-6
    max_grad_norm=1000
    gradient_accumulation_steps=1
    num_warmup_steps=0
    batch_scheduler=True
    print_freq=100
    use_apex=False
    output_dir = './'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = CFG.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)  
    

df = pd.read_csv(CFG.data_path)


def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

#load tokenizer
try: # load pretrained tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    tokenizer = AutoTokenizer.from_pretrained(CFG.pretrained_model_name_or_path, return_tensors='pt')

CFG.tokenizer = tokenizer
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True, return_attention_mask=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    return inputs

    
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df['smiles'].values
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.inputs[item])
        
        return inputs
    
       
class RegressionModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            if 't5' in cfg.pretrained_model_name_or_path:
                self.model = T5EncoderModel.from_pretrained(CFG.pretrained_model_name_or_path)
            else:
                self.model = AutoModel.from_pretrained(CFG.pretrained_model_name_or_path)
        else:
            if 't5' in cfg.model_name_or_path:
                self.model = T5EncoderModel.from_pretrained('sagawa/ZINC-t5')
            else:
                self.model = AutoModel.from_config(self.config)
        self.model.resize_token_embeddings(len(cfg.tokenizer))
        self.fc_dropout1 = nn.Dropout(cfg.fc_dropout)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_dropout2 = nn.Dropout(cfg.fc_dropout)
        self.fc2 = nn.Linear(self.config.hidden_size, 2)
        
    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        output = self.fc1(self.fc_dropout1(last_hidden_states)[:, 0, :].view(-1, self.config.hidden_size))
        output = self.fc2(self.fc_dropout2(output))
        output = F.softmax(output, dim=1)
        return output

    
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


valid_ds = pd.read_csv('../../all_data_for_classification.csv')
valid_ds = valid_ds[valid_ds['target'] == 5].reset_index(drop=True)
CFG.model_name_or_path = 't5'
valid_dataset = TestDataset(CFG, valid_ds)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)
model = RegressionModel(CFG, config_path='./config.pth', pretrained=False)
state = torch.load('./ZINC-t5_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)

model.to(device)
preds = []
for inputs in valid_loader:
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        y_preds = model(inputs)
        preds += torch.argmax(y_preds, dim=1).tolist()

valid_ds['pred'] = preds
valid_ds.to_csv('nodata-prediction.csv', index=False)