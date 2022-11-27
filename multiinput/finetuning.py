import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import tokenizers
import transformers
from transformers import AutoTokenizer, EncoderDecoderModel, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import sentencepiece
import argparse
from sklearn.model_selection import train_test_split
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action='store_true', default=False, required=False)
    parser.add_argument("--epochs", type=int, default=3, required=False)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--max_len", type=int, default=128, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", required=False)
    parser.add_argument("--save_strategy", type=str, default="epoch", required=False)
    parser.add_argument("--logging_strategy", type=str, default="epoch", required=False)
    parser.add_argument("--save_total_limit", type=int, default=2, required=False)
    parser.add_argument("--fp16", action='store_true', default=False, required=False)
    parser.add_argument("--disable_tqdm", action="store_true", default=False, required=False)
    parser.add_argument("--multitask", action="store_true", default=False, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    return parser.parse_args()
    
CFG = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)  
    

# if CFG.dataset_name:
#     if CFG.debug:
#         dataset = load_dataset(CFG.dataset_name)
#         dataset['train'] = datasets.Dataset.from_dict(dataset["train"][:100])
#         dataset['validation'] = datasets.Dataset.from_dict(dataset["validation"][:100])
#     else:
#         dataset = load_dataset(CFG.dataset_name)
# else:
#     if CFG.debug:
#         train = pd.read_csv(CFG.data_path + 'all_ord_reaction_uniq_canonicalized-train.csv')[:100]
#         test = pd.read_csv(CFG.data_path + 'all_ord_reaction_uniq_canonicalized-valid.csv')[:100]
#         train.to_csv(CFG.data_path + 'ord-train-debug.csv', index=False)
#         test.to_csv(CFG.data_path + 'ord-test-debug.csv', index=False)
#         data_files = {'train': CFG.data_path + 'ord-train-debug.csv', 'validation': CFG.data_path + 'ord-test-debug.csv'}
#         dataset = load_dataset('csv', data_files=data_files)
#     else:
#         train = pd.read_csv(CFG.data_path + 'all_ord_reaction_uniq_canonicalized-train.csv')
#         test = pd.read_csv(CFG.data_path + 'all_ord_reaction_uniq_canonicalized-valid.csv')
#         data_files = {'train': CFG.data_path + 'all_ord_reaction_uniq_canonicalized-train.csv', 'validation': CFG.data_path + 'all_ord_reaction_uniq_canonicalized-valid.csv'}
#         dataset = load_dataset('csv', data_files=data_files)
        



# df = pd.read_csv(CFG.data_path)
# df = df[df['NoData'].isna()]
# df_nodata = pd.read_csv('../compound-classification/reconstructed.csv')
# df = pd.concat([df, df_nodata]).reset_index(drop=True)
# df = df[~df['PRODUCT'].isna()]
# for col in ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData','PRODUCT', 'YIELD', 'TEMP']:
#     df[col] = df[col].fillna(' ')
# df['TEMP'] = df['TEMP'].apply(lambda x: str(x))

df = pd.read_csv(CFG.data_path)
df = df[~df['PRODUCT'].isna()]
for col in ['CATALYST', 'REACTANT', 'REAGENT', 'SOLVENT', 'INTERNAL_STANDARD', 'NoData','PRODUCT', 'YIELD', 'TEMP']:
    df[col] = df[col].fillna(' ')
df['TEMP'] = df['TEMP'].apply(lambda x: str(x))


# reactantとcatalyst========================================================================
print(len(df), flush=True)
df = df[df['REACTANT'] != ' ']
print(len(df), flush=True)
df = df[['REACTANT', 'PRODUCT', 'CATALYST', 'REAGENT', 'SOLVENT']].drop_duplicates().reset_index(drop=True)
print(len(df), flush=True)
df = df.iloc[df[['REACTANT', 'CATALYST', 'REAGENT', 'SOLVENT']].drop_duplicates().index].reset_index(drop=True)
print(len(df), flush=True)

def clean(row):
    row = row.replace('. ', '').replace(' .', '').replace('  ', ' ')
    return row
df['REAGENT'] = df['CATALYST'] + '.' + df['REAGENT'] + '.' + df['SOLVENT']
df['REAGENT'] = df['REAGENT'].apply(lambda x: clean(x))
from rdkit import Chem
def canonicalize(mol):
    mol = Chem.MolToSmiles(Chem.MolFromSmiles(mol),True)
    return mol
df['REAGENT'] = df['REAGENT'].apply(lambda x: canonicalize(x) if x != ' ' else ' ')


df['input'] = 'REACTANT:' + df['REACTANT'] + 'REAGENT:' + df['REAGENT']
# ========================================================================

lens = df['input'].apply(lambda x: len(x))
df = df[lens <= 512]

train, test = train_test_split(df, test_size=int(len(df)*0.1))
train, valid = train_test_split(train, test_size=int(len(df)*0.1))


if CFG.debug:
    train = train[:int(len(train)/400)].reset_index(drop=True)
    print('len(train):', len(train), flush=True)
    valid = valid[:int(len(valid)/40)].reset_index(drop=True)
    
    
train[['input', 'PRODUCT']].to_csv('../../multi-input-train.csv', index=False)
valid[['input', 'PRODUCT']].to_csv('../../multi-input-valid.csv', index=False)
test[['input', 'PRODUCT']].to_csv('../../multi-input-test.csv', index=False)

nodata = pd.read_csv('/data2/sagawa/transformer-chemical-reaction-prediciton/compound-classification/reconstructed.csv')
nodata = nodata[~nodata['REACTANT'].isna()]
for col in ['REAGENT']:
    nodata[col] = nodata[col].fillna(' ')
nodata['input'] = 'REACTANT:' + nodata['REACTANT'] + 'REAGENT:' + nodata['REAGENT']
train = pd.concat([train[['input', 'PRODUCT']], nodata[['input', 'PRODUCT']]]).reset_index(drop=True)


dataset = DatasetDict({'train': Dataset.from_pandas(train[['input', 'PRODUCT']]), 'validation': Dataset.from_pandas(valid[['input', 'PRODUCT']])})


def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['PRODUCT']
    model_inputs = tokenizer(inputs, max_length=CFG.max_len, truncation=True)
    labels = tokenizer(targets, max_length=CFG.max_len, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def compute_metrics(eval_preds):
    metric = load_metric('sacrebleu')
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {'bleu': result['score']}


#load tokenizer
try: # load pretrained tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors='pt')
except: # load pretrained tokenizer from huggingface model hub
    tokenizer = AutoTokenizer.from_pretrained(CFG.pretrained_model_name_or_path, return_tensors='pt')
tokenizer.add_tokens('.')
tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + ['REACTANT:', 'REAGENT:']})


#load model
if CFG.model == 't5':
    try: # load pretrained model from local directory
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), from_flax=True)
    except: # load pretrained model from huggingface model hub
        model = AutoModelForSeq2SeqLM.from_pretrained(CFG.pretrained_model_name_or_path, from_flax=True)
    model.resize_token_embeddings(len(tokenizer))
elif CFG.model == 'deberta':
    try: # load pretrained model from local directory
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), 'roberta-large')
    except: # load pretrained model from huggingface model hub
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), 'roberta-large')
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.decoder.resize_token_embeddings(len(tokenizer))
    config_encoder = model.config.encoder
    config_decoder = model.config.decoder
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    load_from_cache_file=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    CFG.model,
    evaluation_strategy=CFG.evaluation_strategy,
    save_strategy=CFG.save_strategy,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    weight_decay=CFG.weight_decay,
    save_total_limit=CFG.save_total_limit,
    num_train_epochs=CFG.epochs,
    predict_with_generate=True,
    fp16=CFG.fp16,
    disable_tqdm=CFG.disable_tqdm,
    push_to_hub=False,
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./best_model')

