import re, gc, glob, io, tokenize, markdown
import pandas as pd
import numpy as np
import torch
import configuration as configuration
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm.auto import tqdm
from torch import Tensor


def kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ KFold """
    fold = KFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df)):
        df.loc[vx, "fold"] = int(num)
    return df


def stratified_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ GroupKFold """
    fold = StratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(X=df, y=df['prompt_id'])):
        df.loc[vx, "fold"] = int(num)
    return df


def mls_kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ Multilabel Stratified KFold """
    tmp_df = df.copy()
    y = pd.get_dummies(data=tmp_df.iloc[:, 2:8], columns=tmp_df.columns[2:8])
    fold = MultilabelStratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    for num, (tx, vx) in enumerate(fold.split(X=df, y=y)):
        df.loc[vx, "fold"] = int(num)
    del tmp_df
    gc.collect()
    return df


def add_target_token(cfg: configuration.CFG, token: str) -> None:
    """
    Add special token to pretrained tokenizer
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token: str, special token to add
    """
    special_token = token
    special_tokens_dict = {'additional_special_tokens': [f'{special_token}']}
    cfg.tokenizer.add_special_tokens(special_tokens_dict)
    tar_token_id = cfg.tokenizer(f'{special_token}', add_special_tokens=False)['input_ids'][0]

    setattr(cfg.tokenizer, 'tar_token', f'{special_token}')
    setattr(cfg.tokenizer, 'tar_token_id', tar_token_id)
    cfg.tokenizer.save_pretrained(f'{cfg.checkpoint_dir}/tokenizer/')


def add_common_token(cfg: configuration.CFG, token: str) -> None:
    """
    Add special token to pretrained tokenizer
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token: str, special token to add
    """
    special_token = token
    special_tokens_dict = {'additional_special_tokens': [f'{special_token}']}
    cfg.tokenizer.add_special_tokens(special_tokens_dict)
    com_token_id = cfg.tokenizer(f'{special_token}', add_special_tokens=False)['input_ids'][0]

    setattr(cfg.tokenizer, 'common_token', f'{special_token}')
    setattr(cfg.tokenizer, 'common_token_id', com_token_id)
    cfg.tokenizer.save_pretrained(f'{cfg.checkpoint_dir}/tokenizer/')


def tokenizing(cfg: configuration.CFG, text: str) -> any:
    """
    Preprocess text for LLM Input
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        max_length=cfg.max_len,
        padding='max_length',
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,  # later, we will add ourselves
    )
    for k, v in inputs.items():
        # inputs[k] = torch.as_tensor(v)
        inputs[k] = torch.tensor(v)
    return inputs


def adjust_sequences(sequences: list, max_len: int):
    """
    Similar to dynamic padding concept
    Append slicing index from original, because original source code is implemented weired
    So it generates some problem for applying very longer sequence
    Add -1 value to slicing index, so we can get result what we want
    Args:
        sequences: list of each cell's token sequence in one unique notebook id, must pass tokenized sequence input_ids
        => sequences = [[1,2,3,4,5,6], [1,2,3,4,5,6], ... , [1,2,3,4,5]]
        max_len: max length of sequence into LLM Embedding Layer, default is 2048 for DeBERTa-V3-Large
    Reference:
         https://github.com/louis-she/ai4code/blob/master/ai4code/utils.py#L70
    """
    length_of_seqs = [len(seq) for seq in sequences]
    total_len = sum(length_of_seqs)
    cut_off = total_len - max_len
    if cut_off <= 0:
        return sequences, length_of_seqs

    for _ in range(cut_off):
        max_index = length_of_seqs.index(max(length_of_seqs))
        length_of_seqs[max_index] -= 1
    sequences = [sequences[i][:l-1] for i, l in enumerate(length_of_seqs)]
    return sequences, length_of_seqs


def subsequent_tokenizing(cfg: configuration.CFG, text: str) -> any:
    """
    Tokenize input sentence to longer sequence than common tokenizing
    Append padding strategy NOT Apply same max length, similar concept to dynamic padding
    Truncate longer sequence to match LLM max sequence
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/343714
        https://github.com/louis-she/ai4code/blob/master/tests/test_utils.py#L6
    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        # max_length=64,
        padding=False,
        # truncation=True,
        return_tensors=None,
        add_special_tokens=False,  # No need to special token to subsequent text sequence
    )
    return inputs['input_ids']


def find_index(x: Tensor, value: Tensor) -> Tensor:
    """
    Method for find some tensor element's index
    Args:
        x: tensor object, which is contained whole tensor elements
        value: element that you want to find index
    """
    tensor_index = torch.where(x == value)
    return tensor_index


def subsequent_decode(cfg: configuration.CFG, token_list: list) -> any:
    """
    Return decoded text from subsequent_tokenizing & adjust_sequences
    For making prompt text
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token_list: token list from subsequent_tokenizing & adjust_sequences
    """
    output = cfg.tokenizer.decode(token_list)
    return output


def sequence_length(cfg: configuration.CFG, text_list: list) -> list:
    """ Get sequence length of all text data for checking statistics value """
    length_list = []
    for text in tqdm(text_list):
        tmp_text = tokenizing(cfg, text)['attention_mask']
        length_list.append(tmp_text.count(1))
    return length_list


def check_null(df: pd.DataFrame) -> pd.Series:
    """ check if input dataframe has null type object...etc """
    return df.isnull().sum()


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(data_path)
    return df


def create_word_normalizer():
    """
    Create a function that normalizes a word.
    """
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def normalize(word):
        w = word.lower()
        w = lemmatizer.lemmatize(w)
        w = ps.stem(w)
        return w
    return normalize


def __normalize_words(titles: list) -> list:
    """
    Normalize a list of words
    1) Remove stop words
    2) Apply Porter Stemmer, Lemmatizer
    """
    stop_words = set(stopwords.words('english'))
    normalizer = create_word_normalizer()
    titles = [normalizer(t) for t in titles if t not in stop_words]
    return titles


def normalize_words(words: np.ndarray, unique=True) -> list:
    """
    Normalize a list of words
    1) Apply __normalize_word function
    2) Apply Regular Expression to remove special characters
    """
    if type(words) is str:
        words = [words]
    sep_re = r'[\s\(\){}\[\];,\.]+'
    num_re = r'\d'
    words = re.split(sep_re, ' '.join(words).lower())
    words = [w for w in words if len(w) >= 3 and not re.match(num_re, w)]
    if unique:
        words = list(set(words))
        words = set(__normalize_words(words))
    else:
        words = __normalize_words(words)
    return words


