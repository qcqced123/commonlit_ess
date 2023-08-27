import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import configuration
from preprocessing import add_special_token, tokenizing, adjust_sequences
from preprocessing import subsequent_tokenizing, subsequent_decode, sequence_length


class OneToOneDataset(Dataset):
    """
    For Supervised Learning Pipeline, making "type 1" prompt sentence for LLMs Inputs
    This class have 2 functions
        1) apply preprocessing for mis-spelling words in prompt texts & summaries texts (maybe later applying)
        2) make prompt sentence with no padding
    Args:
        cfg: config module from configuration.py
        df: train/inference dataframe

    """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int):
        p_ids = torch.from_numpy(self.df.prompt_ids.to_numpy())
        p_titles = torch.from_numpy(self.df.prompt_title.to_numpy())
        p_texts = torch.from_numpy(self.df.prompt_text.to_numpy())
        p_questions = torch.from_numpy(self.df.prompt_question.to_numpy())


class OneToManyDataset(Dataset):
    """
    For Supervised Learning Pipeline, making "type 2" prompt sentence for LLMs Inputs
    This class have 4 functions
        1) apply preprocessing for mis-spelling words in prompt texts & summaries texts (maybe later applying)
        2) make prompt sentence with no padding
        3) split original data schema into several subsets
        4) random shuffle for each target instance (summaries texts, labels)
    Args:
        cfg: config module from configuration.py
        df: train/inference dataframe
    """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int):
        p_ids = torch.from_numpy(self.df.prompt_ids.to_numpy())
        p_titles = torch.from_numpy(self.df.prompt_title.to_numpy())
        p_texts = torch.from_numpy(self.df.prompt_text.to_numpy())
        p_questions = torch.from_numpy(self.df.prompt_question.to_numpy())
