
import argparse
import datetime as dt
import glob
import multiprocessing as mp
import os
from typing import Tuple, List
import logging
import urllib.request
import tarfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import sklearn.metrics
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import transformers

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# transformers.logger.setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

def make_dataset() -> Tuple[np.array, np.array, int]:
    path_pkl = 'dataset.pkl'
    if os.path.isfile(path_pkl):
        logger.debug('Load data from cache')
        df = pd.read_pickle(path_pkl)
    else:
        logger.debug('Newly download data')
        file_name, headers = urllib.request.urlretrieve('https://www.rondhuit.com/download/ldcc-20140209.tar.gz')
        with tarfile.open(file_name) as tar:
            tar.extractall()
        path_data = './text/'
        list_filename = list(filter(lambda x: x.split('/')[-1] not in ['CHANGES.txt', 'README.txt', 'LICENSE.txt'], glob.glob(os.path.join(path_data, '*/*.txt'))))

        list_text = []
        list_category = []
        for filename in list_filename:
            with open(filename, 'r') as f:
                text = f.read()
            text = ''.join(text.split('\n')[3:])
            list_text.append(text)
            list_category.append(filename.split('/')[-2])
            
        list_uniq_category = np.unique(list_category).tolist()
        list_label = [list_uniq_category.index(x) for x in list_category]
        df = pd.DataFrame({
            'text': list_text,
            'label': list_label
        })
        df.to_pickle(path_pkl)
    arr_text = df['text'].values
    arr_label = df['label'].values
    return arr_text, arr_label, df['label'].nunique()

def main(path:str, path_vocab:str, max_length:int, neologd:bool, layers:int) -> None:
    random_state = 42
    batch_size = 128
    lr = 3e-3
    max_epoch = 40
    device = torch.device('cuda')
    logger.debug(f'length: {max_length}, batch: {batch_size}, lr: {lr}, layers: {layers}')
    
    arr_text, arr_label, num_labels = make_dataset()
    global TOKENIZER
    if path_vocab != '':
        if os.path.isfile(path_vocab) and not neologd:
            logger.debug(f'vocab path: {path_vocab}')
            TOKENIZER = transformers.BertJapaneseTokenizer(
                path_vocab,
                do_lower_case = False,
                word_tokenizer_type = "mecab",
                subword_tokenizer_type = "wordpiece",
                tokenize_chinese_chars = False,
                # mecab_kwargs = {'mecab_dic': mecab_dic_type}
            )
        elif neologd:
            logger.debug(f'vocab path: {path_vocab} (with neologd)')
            TOKENIZER = transformers.BertJapaneseTokenizer.from_pretrained(path, mecab_kwargs={"mecab_option": "-d /home/b2019msuzuki/local/lib/mecab/dic/mecab-ipadic-neologd"})
        else:
            TOKENIZER = transformers.BertJapaneseTokenizer.from_pretrained(path_vocab)
    else:
        TOKENIZER = transformers.BertJapaneseTokenizer.from_pretrained(path)
    with mp.Pool(os.cpu_count()) as pool: #tokenize
        mp_task = [pool.apply_async(mp_tokenize, (text, max_length)) for text in arr_text]
        tokens = np.array([f.get() for f in mp_task])
    # tokens = np.array([tokenizer.encode(x, padding='max_length', truncation=True, max_length=max_length) for x in arr_text])
    skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    list_valid_kfold_acc, list_test_kfold_acc = [], []

    logger.debug(f'model: {path}')
    for i, (traval_idx, test_idx) in enumerate(skf.split(tokens, arr_label)):
        traval_tokens = tokens[traval_idx]
        test_tokens = tokens[test_idx]
        traval_labels = arr_label[traval_idx]
        test_labels = arr_label[test_idx]
        train_tokens, valid_tokens, train_labels, valid_labels = train_test_split(traval_tokens, traval_labels, test_size=0.15, random_state=random_state, shuffle=True)
        train_tokens = torch.tensor(train_tokens).to(device)
        valid_tokens = torch.tensor(valid_tokens).to(device)
        test_tokens = torch.tensor(test_tokens).to(device)
        train_labels = torch.tensor(train_labels).to(device)
        valid_labels = torch.tensor(valid_labels).to(device)
        test_labels  = torch.tensor(test_labels).to(device)
        train_loader = data.DataLoader(data.TensorDataset(
            train_tokens, train_labels
        ), batch_size=batch_size, shuffle=True)
        valid_loader = data.DataLoader(data.TensorDataset(
            valid_tokens, valid_labels
        ), batch_size=batch_size, shuffle=False)
        test_loader = data.DataLoader(data.TensorDataset(
            test_tokens, test_labels
        ), batch_size=batch_size, shuffle=False)

        if 'bert' in path:
            config = transformers.BertConfig.from_pretrained(path)
            config.num_labels = num_labels
            net = transformers.BertForSequenceClassification.from_pretrained(path, config=config)
        elif 'electra' in path:
            config = transformers.ElectraConfig.from_pretrained(path)
            config.num_labels = num_labels
            net = transformers.ElectraForSequenceClassification.from_pretrained(path, config=config)
        else:
            raise ValueError('path must contain bert or electra')
        net = net.to(device)
        if 'bert' in path:
            # 一旦全部のパラメータのrequires_gradをFalseで更新
            for name, param in net.named_parameters():
                param.requires_grad = False
            for j in range(1, layers+1):
                for name, param in net.bert.encoder.layer[-j].named_parameters():
                    param.requires_grad = True
            # Bert poolerレイヤのrequires_gradをTrueで更新
            for name, param in net.bert.pooler.named_parameters():
                param.requires_grad = True
            # 最後のclassificationレイヤのrequires_gradをTrueで更新
            for name, param in net.classifier.named_parameters():
                param.requires_grad = True
        elif 'electra' in path:
            # 一旦全部のパラメータのrequires_gradをFalseで更新
            for name, param in net.named_parameters():
                param.requires_grad = False
            for j in range(1, layers+1):
                for name, param in net.electra.encoder.layer[-j].named_parameters():
                    param.requires_grad = True
            # electra poolerレイヤのrequires_gradをTrueで更新
            for name, param in net.classifier.dense.named_parameters():
                param.requires_grad = True
            # 最後のclassificationレイヤのrequires_gradをTrueで更新
            for name, param in net.classifier.out_proj.named_parameters():
                param.requires_grad = True
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        def valid(loader):
            real, pred = [], [] # real: answer
            net.eval()
            with torch.no_grad():
                for inputs, labels in loader:
                    # All options are default
                    loss, logits = net(inputs,labels=labels)[:2]
                    real.extend(labels.tolist())
                    pred.extend(logits.max(1)[1].tolist())
            acc = sklearn.metrics.accuracy_score(real, pred)
            return acc
        
        list_valid_acc, list_test_acc = [], []
        for j in range(1, max_epoch+1):
            net.train()
            train_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                # All options are default
                loss, logits = net(inputs, labels=labels)[:2]
                train_loss += loss.item() * labels.size(0)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # valid
            valid_acc = valid(valid_loader)
            list_valid_acc.append(valid_acc)
            # test
            test_acc = valid(test_loader)
            list_test_acc.append(test_acc)

            logger.debug(f"[{i+1}/{j:02}] TrL: {train_loss:.3g}, Vacc: {valid_acc:.4f}, Tacc: {test_acc:.4f}")
            # end of train&valid
        valid_acc = max(list_valid_acc)
        test_acc = list_test_acc[np.argmax(list_valid_acc)]
        logger.debug(f'Trial[{i+1}] Valid Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}')
        list_valid_kfold_acc.append(valid_acc)
        list_test_kfold_acc.append(test_acc)
    logger.debug(f'Average Valid Acc: {np.mean(list_valid_kfold_acc):.5f}')
    logger.debug(f'Average Test Acc: {np.mean(list_test_kfold_acc):.5f}')


def mp_tokenize(text, max_length):
    return TOKENIZER.encode(text, padding='max_length', truncation=True, max_length=max_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--path_vocab', type=str, default='')
    parser.add_argument('--layers', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--neologd', action='store_true')
    args = parser.parse_args()

    # logger
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ## create file handler which logs even DEBUG messages
    yymmdd_hhmm = dt.datetime.now().strftime('%Y%m%d_%H%M')
    fh = logging.FileHandler('log_' + yymmdd_hhmm + '.txt')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        # fmt='%(asctime)s %(levelname)s: %(message)s', 
        fmt='%(asctime)s: %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    ## create console handler with a INFO log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        # fmt='%(asctime)s %(levelname)s: %(message)s', 
        fmt='%(asctime)s: %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main(path=args.path, path_vocab=args.path_vocab, max_length=args.max_length, neologd=args.neologd, layers=args.layers)
