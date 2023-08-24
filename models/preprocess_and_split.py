from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import numpy as np
import shutil
import json
import os

def filter_data(data):
    data[['PP', 'SP', 'MC']] = data[['PP', 'SP', 'MC']].astype(int)
    allowed_combinations = [
        (0, 6, 0), (0, 0, 6), (6, 0, 0), (4, 2, 0), (2, 4, 0),
        (0, 2, 4), (0, 4, 2), (2, 0, 4), (1, 5, 0), (0, 5, 1),
        (0, 1, 5), (4, 0, 2), (5, 1, 0), (1, 0, 5), (5, 0, 1)
    ]
    data = data[data[['PP', 'SP', 'MC']].apply(tuple, axis=1).isin(allowed_combinations)]
    data.reset_index(drop=True, inplace=True)
    data.reindex()
    return data

def binarize_data(data):
    data['PP'] = data['PP'].apply(lambda x: 1 if x > 3 else 0)
    data['SP'] = data['SP'].apply(lambda x: 1 if x > 3 else 0)
    data['MC'] = data['MC'].apply(lambda x: 1 if x > 3 else 0)
    return data

def add_label_column(data):
    def label(row):
        if row['PP'] == 1 and row['SP'] == 0 and row['MC'] == 0:
            return 0
        elif row['PP'] == 0 and row['SP'] == 1 and row['MC'] == 0:
            return 1
        elif row['PP'] == 0 and row['SP'] == 0 and row['MC'] == 1:
            return 2
        else:
            return None
    data['label'] = data.apply(label, axis=1)
    return data

def split_for_one_vs_rest(train):
    train_df = pd.DataFrame(train)
    pp_train = train_df[train_df['PP'] == 1].reset_index(drop=True)
    sp_train = train_df[train_df['SP'] == 1].reset_index(drop=True)
    mc_train = train_df[train_df['MC'] == 1].reset_index(drop=True)
    n_pp_1 = pp_train['PP'].sum()
    pp_train_0 = train_df[train_df['PP'] == 0].sample(n=n_pp_1, replace=True)
    n_sp_1 = sp_train['SP'].sum()
    sp_train_0 = train_df[train_df['SP'] == 0].sample(n=n_sp_1, replace=True)
    n_mc_1 = mc_train['MC'].sum()
    mc_train_0 = train_df[train_df['MC'] == 0].sample(n=n_mc_1, replace=True)
    pp_train = pd.concat([pp_train_0, pp_train], axis=0).reset_index(drop=True)
    sp_train = pd.concat([sp_train_0, sp_train], axis=0).reset_index(drop=True)
    mc_train = pd.concat([mc_train_0, mc_train], axis=0).reset_index(drop=True)
    pp_train = pp_train.to_dict(orient='records')
    sp_train = sp_train.to_dict(orient='records')
    mc_train = mc_train.to_dict(orient='records')
    return pp_train, sp_train, mc_train

def remove_irrelevant_columns(datasets_dir):
    updated_datasets = []
    for fold in os.listdir(datasets_dir):
        if not fold.startswith('cv_fold_'):
            continue
        fold_dir = os.path.join(datasets_dir, fold)
        for dataset_file in os.listdir(fold_dir):
            dataset_name = dataset_file.split('.')[0]
            dataset_path = os.path.join(fold_dir, dataset_file)
            with open(dataset_path, 'r') as f:
                dataset_json = f.read()
            dataset_df = pd.read_json(dataset_json, orient='records', lines=True)
            if "pp" in dataset_name:
                dataset_df = dataset_df.drop(["SP", "MC"], axis=1)
            elif "sp" in dataset_name:
                dataset_df = dataset_df.drop(["PP", "MC"], axis=1)
            elif "mc" in dataset_name:
                dataset_df = dataset_df.drop(["PP", "SP"], axis=1)
            updated_dataset_json = dataset_df.to_json(orient='records', lines=True)
            with open(dataset_path, 'w') as f:
                f.write(updated_dataset_json)
            updated_datasets.append((dataset_name, updated_dataset_json))
    return updated_datasets

def count_instances_and_labels(cv_dir, num_folds=10, output_file="output.txt"):
    with open(output_file, "w") as f:
        for i in range(1, num_folds + 1):
            f.write(f"Fold {i}:\n")
            dir_path = os.path.join(cv_dir, f'cv_fold_{i}')
            for partition in ['train', 'val', 'test', 'pp_train', 'sp_train', 'mc_train']:
                file_path = os.path.join(dir_path, f'{partition}.json')
                with open(file_path, 'r') as json_file:
                    data = [json.loads(line) for line in json_file.readlines()]
                df = pd.DataFrame(data)
                total_count = len(df)
                f.write(f"  {partition.capitalize()} set:\n")
                f.write(f"    Total instances: {total_count}\n")
                for col in ['PP', 'SP', 'MC']:
                    if col in df.columns:
                        col_counts = df[col].value_counts().to_dict()
                        f.write(f"    {col} counts: {col_counts}\n")

def check_duplicate_ids(cv_dir, num_folds=10):
    for i in range(1, num_folds + 1):
        print(f"Checking Fold {i} for duplicate IDs...")
        dir_path = os.path.join(cv_dir, f'cv_fold_{i}')
        all_ids = []
        for partition in ['train', 'val', 'test']:
            file_path = os.path.join(dir_path, f'{partition}.json')
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f.readlines()]
            df = pd.DataFrame(data)
            ids = df['Segment ID'].tolist()
            all_ids.extend(ids)
        duplicate_ids = set([x for x in all_ids if all_ids.count(x) > 1])
        if duplicate_ids:
            print(f"  Found duplicate IDs in fold {i}: {duplicate_ids}")
        else:
            print(f"  No duplicate IDs found in fold {i}")

def split_data(data_path, output_dir='/content'):
    data = pd.read_excel(data_path)
    data = filter_data(data)
    data = binarize_data(data)
    data = add_label_column(data)
    df = data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df['label']
    n_splits = 10

    train_val, test = train_test_split(df, test_size=0.1, stratify=y, random_state=42)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_val_idx, val_idx) in enumerate(skf.split(train_val, train_val['label'])):
        train = train_val.iloc[train_val_idx]
        val = train_val.iloc[val_idx]

        base_path = f'{output_dir}/cv_fold_{fold + 1}'
        os.makedirs(base_path, exist_ok=True)
        datasets = [('train', train), ('val', val), ('test', test)]
        for dataset_name, dataset in datasets:
            file_path = os.path.join(base_path, f'{dataset_name}.json')
            dataset_df = pd.DataFrame(dataset)
            dataset_df.to_json(file_path, orient='records', lines=True)

        pp_train, sp_train, mc_train = split_for_one_vs_rest(train)
        one_vs_rest_datasets = [('pp_train', pp_train), ('sp_train', sp_train), ('mc_train', mc_train)]
        for dataset_name, dataset in one_vs_rest_datasets:
            file_path = os.path.join(base_path, f'{dataset_name}.json')
            dataset_df = pd.DataFrame(dataset)
            dataset_df.to_json(file_path, orient='records', lines=True)

    updated_datasets = remove_irrelevant_columns(output_dir)
    output_file = f"{output_dir}/cv_folds_statistics.txt"
    count_instances_and_labels(output_dir, output_file=output_file)
    check_duplicate_ids(output_dir)