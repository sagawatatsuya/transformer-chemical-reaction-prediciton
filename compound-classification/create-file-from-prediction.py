import os
import gc
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import pickle
import time
import math
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed=42)  


df = pd.read_csv('nodata-prediction.csv')
dic = {}
for idx, row in df.iterrows():
    dic[row['smiles']] = row['pred']
    
df_all = pd.read_csv('../../all_ord_reaction_uniq_with_attr_v3.tsv')
df_all = df_all[~df_all['PRODUCT'].isna()]
df_all = df_all[df_all['REACTANT'].isna()]

dfs = []
name2num = {'REAGENT': 0,
 'REACTANT': 1}
names = ['REAGENT', 'REACTANT']
for idx, row in df_all.iterrows():
    lis = []
    if type(row['NoData']) != str:
        lis.append(row[names].values)
    else:
        for name in names:
            if type(row[name]) == str:
                lis.append([row[name]])
            else:
                lis.append([])
        for smile in row['NoData'].split('.'):
            num = dic[smile]
            lis[num].append(smile)
        lis = ['.'.join(smiles) for smiles in lis]
    dfs.append(lis)

df_after = pd.DataFrame(np.array(dfs).reshape(-1, 2), columns=names)

def canonicalize(smi):
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi),True)
    return smi

df_after[['PRODUCT', 'YIELD', 'TEMP']] = df_all[['PRODUCT', 'YIELD', 'TEMP']]

df_after.to_csv('reconstructed-nodata.csv', index=False)
df = pd.read_csv('reconstructed-nodata.csv')


# df['CATALYST'] = df['CATALYST'].apply(lambda x: canonicalize(x) if type(x) == str else None)
df['REACTANT'] = df['REACTANT'].apply(lambda x: canonicalize(x) if type(x) == str else None)
# df['REAGENT'] = df['REAGENT'].apply(lambda x: canonicalize(x) if type(x) == str else None)
# df['SOLVENT'] = df['SOLVENT'].apply(lambda x: canonicalize(x) if type(x) == str else None)
# df['INTERNAL_STANDARD'] = df['INTERNAL_STANDARD'].apply(lambda x: canonicalize(x) if type(x) == str else None)

df.to_csv('reconstructed-nodata.csv', index=False)
print('finish')