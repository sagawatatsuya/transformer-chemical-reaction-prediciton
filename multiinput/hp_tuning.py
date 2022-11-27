import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import tokenizers
from transformers import AutoTokenizer, AutoModel, AutoConfig, EncoderDecoderModel, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import datasets
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import sentencepiece
import argparse
import optuna
from transformers.trainer_utils import HPSearchBackend, default_hp_space, PREFIX_CHECKPOINT_DIR, default_compute_objective, BestRun
import gc
from sklearn.model_selection import train_test_split
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default='output',required=False)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action='store_true', default=False, required=False)
    parser.add_argument("--epochs", type=int, default=2, required=False)
    parser.add_argument("--n_trials", type=int, default=10, required=True)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument("--batch_size", type=int, default=5, required=False)
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", required=False)
    parser.add_argument("--logging_strategy", type=str, default="epoch", required=False)
    parser.add_argument("--fp16", action='store_true', default=False, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()
    
CFG = parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists(CFG.output_dir):
    os.makedirs(CFG.output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=CFG.seed)


# https://github.com/huggingface/transformers/issues/13019
def run_hp_search_optuna(trainer, n_trials, direction, **kwargs):
    import optuna
    def _objective(trial, checkpoint_dir=None):
        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        #################
        ## UPDATES START
        #################
        if not checkpoint:
            # free GPU memory
            del trainer.model
            gc.collect()
            torch.cuda.empty_cache()
        trainer.objective = None
        trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params)

def hyperparameter_search(trainer, n_trials, hp_space = None, compute_objective = None, direction = "minimize", hp_name = None, **kwargs):
    trainer.hp_search_backend = HPSearchBackend.OPTUNA
    trainer.hp_space = default_hp_space[HPSearchBackend.OPTUNA] if hp_space is None else hp_space
    trainer.hp_name = hp_name
    trainer.compute_objective = default_compute_objective if compute_objective is None else compute_objective
    best_run = run_hp_search_optuna(trainer, n_trials, direction, **kwargs)
    trainer.hp_search_backend = None
    return best_run


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
df['input'] = 'REACTANT:' + df['REACTANT'] + 'CATALYST:' + df['CATALYST'] + 'REAGENT:' + df['REAGENT'] + 'SOLVENT:' + df['SOLVENT'] 
# ========================================================================

lens = df['input'].apply(lambda x: len(x))
df = df[lens <= 512]

train, test = train_test_split(df, test_size=int(len(df)*0.1))
train, valid = train_test_split(train, test_size=int(len(df)*0.1))


train = train[:int(len(train)/400)].reset_index(drop=True)
valid = valid[:int(len(valid)/40)].reset_index(drop=True)
    

dataset = DatasetDict({'train': Dataset.from_pandas(train[['input', 'PRODUCT']]), 'validation': Dataset.from_pandas(valid[['input', 'PRODUCT']])})

    
    

def hp_tuning(cfg):

    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['PRODUCT']
        model_inputs = tokenizer(inputs, max_length=cfg.max_len, truncation=True)
        labels = tokenizer(targets, max_length=cfg.max_len, truncation=True)
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
    
    # load tokenizer
    try: # load pretrained tokenizer from local directory
        tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(CFG.pretrained_model_name_or_path), return_tensors='pt')
    except: # load pretrained tokenizer from huggingface model hub
        tokenizer = AutoTokenizer.from_pretrained(CFG.pretrained_model_name_or_path, return_tensors='pt')
    tokenizer.add_tokens('.')
    tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + ['CATALYST:', 'REACTANT:', 'REAGENT:', 'SOLVENT:', 'NoData:','PRODUCT:']})


    def get_model():
        if cfg.model == 't5':
            try: # load pretrained model from local directory
                model = AutoModelForSeq2SeqLM.from_pretrained(os.path.abspath(cfg.pretrained_model_name_or_path), from_flax=True)
            except: # load pretrained model from huggingface model hub
                model = AutoModelForSeq2SeqLM.from_pretrained(cfg.pretrained_model_name_or_path, from_flax=True)
            model.resize_token_embeddings(len(tokenizer))
            
        elif cfg.model == 'deberta':
            try: # load pretrained model from local directory
                model = EncoderDecoderModel.from_encoder_decoder_pretrained(os.path.abspath(cfg.pretrained_model_name_or_path), 'roberta-large')
            except: # load pretrained model from huggingface model hub
                model = EncoderDecoderModel.from_encoder_decoder_pretrained(os.path.abspath(cfg.pretrained_model_name_or_path), 'roberta-large')
            model.encoder.resize_token_embeddings(len(tokenizer))
            model.decoder.resize_token_embeddings(len(tokenizer))
            config_encoder = model.config.encoder
            config_decoder = model.config.decoder
            config_decoder.is_decoder = True
            config_decoder.add_cross_attention = True
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            
        return model
    # load model
    model = get_model()

    # tokenize dataset
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        load_from_cache_file=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True, 
        evaluation_strategy=cfg.evaluation_strategy,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.epochs,
        predict_with_generate=True,
        save_total_limit=2,
        fp16=cfg.fp16,
        push_to_hub=False,
        disable_tqdm=True
    )
    
    Seq2SeqTrainer.hyperparameter_search = hyperparameter_search

    trainer = Seq2SeqTrainer(
        model_init=get_model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.1, log=True),
        }

    # start hyperparameter tuning
    param = trainer.hyperparameter_search(
        direction='maximize',
        hp_space=my_hp_space,
        n_trials=cfg.n_trials
    )
    with open('result.txt', 'w') as f:
        f.write(str(param))

if __name__ == "__main__":
    hp_tuning(CFG)