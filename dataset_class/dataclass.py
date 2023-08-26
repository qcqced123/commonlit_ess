import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import configuration
from preprocessing import add_special_token, tokenizing, adjust_sequences
from preprocessing import subsequent_tokenizing, subsequent_decode, sequence_length


class CommonLitDataset(Dataset):
    """
    For Supervised Learning Pipeline
    This class have 2 functions, maybe second function not apply later because of max sequence limitation
        1) make prompt sentence with no padding
        2) random shuffle each data instance
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


