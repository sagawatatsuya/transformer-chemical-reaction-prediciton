import os
import gc
import random
import itertools
import warnings
import logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
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
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import mean_squared_error, r2_score
disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=False)
#     parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="sagawa/ZINC-t5", required=False)
    parser.add_argument("--model", type=str, default="t5", required=False)
    parser.add_argument("--model_name_or_path", type=str, required=False)
#     parser.add_argument("--scaler_path", type=str, default="/data2/sagawa/tcrp-regression-model-archive/10-23-1st-new-metric-reactant-product", required=False)
    parser.add_argument("--debug", action='store_true', default=False, required=False)
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--batch_size", type=int, default=5, required=False)
    parser.add_argument("--fc_dropout", type=float, default=0.1, required=False)
    parser.add_argument("--num_workers", type=int, default=1, required=False)
    parser.add_argument("--output_dir", type=str, default='./', required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    return parser.parse_args()

CFG = parse_args()

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


# with open(CFG.scaler_path + '/scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

    

try: # load pretrained tokenizer from local directory
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path+'/tokenizer', return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path, return_tensors='pt')

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False, truncation=True, return_attention_mask=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    
    return inputs

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.inputs = df['input'].values
        
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
            if 't5' in cfg.model:
                self.model = T5EncoderModel.from_pretrained(CFG.pretrained_model_name_or_path, config=self.config)
            else:
                self.model = AutoModel.from_pretrained(CFG.pretrained_model_name_or_path, config=self.config)
        else:
            if 't5' in cfg.model:
                self.model = T5EncoderModel.from_pretrained('sagawa/ZINC-t5', config=self.config)
            else:
                self.model = AutoModel.from_config(self.config)
        self.model.resize_token_embeddings(len(cfg.tokenizer))
        self.fc_dropout1 = nn.Dropout(cfg.fc_dropout)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_dropout2 = nn.Dropout(cfg.fc_dropout)
        self.fc2 = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        output = self.fc1(self.fc_dropout1(last_hidden_states)[:, 0, :].view(-1, self.config.hidden_size))
        output = self.fc2(self.fc_dropout2(output))
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

model = RegressionModel(CFG, config_path=CFG.model_name_or_path + '/config.pth', pretrained=False)
state = torch.load(CFG.model_name_or_path + '/ZINC-t5_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)


if 'csv' in CFG.data:
    test_ds = pd.read_csv(CFG.data)
    
    if CFG.debug:
        test_ds = test_ds[:500]
        
    test_dataset = TestDataset(CFG, test_ds)
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    prediction = inference_fn(test_loader, model, device)

    test_ds['prediction'] = prediction*100
    test_ds['prediction'] = test_ds['prediction'].clip(0, 100)
#     test_ds['prediction'] = scaler.inverse_transform(test_ds['prediction'].values.reshape(-1, 1))
    test_ds.to_csv(CFG.output_dir + 'yield_prediction_output.csv', index=False)
    
else:
    
    test_ds = pd.DataFrame.from_dict({'input': CFG.data}, orient='index').T
    test_dataset = TestDataset(CFG, test_ds)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    prediction = inference_fn(test_loader, model, device)
    prediction = max(min(prediction[0][0]*100, 100), 0)
    print('yiled: ', prediction)
 