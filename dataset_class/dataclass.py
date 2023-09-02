import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

import configuration
from preprocessing import tokenizing, adjust_sequences
from preprocessing import subsequent_tokenizing, subsequent_decode, sequence_length, find_index


class OneToOneDataset(Dataset):
    """
    For Supervised Learning Pipeline, making "type 1" prompt sentence for LLMs Inputs
    This class have 2 functions
        1) apply preprocessing for mis-spelling words in prompt texts & summaries texts (maybe later applying)
        2) make prompt sentence with no padding (x)
        => this pipeline will be set hyper-params 'max_len' as at least 1920,
        which is maximum value of all instance's prompt sentence length
    Args:
        cfg: config module from configuration.py
        p_df: prompt train dataset, prompts_train.csv
        s_df: prompt train dataset, summaries_train.csv
    """
    def __init__(self, cfg: configuration.CFG, p_df: pd.DataFrame, s_df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.tokenizing = tokenizing
        self.subsequent_tokenizing = subsequent_tokenizing
        self.adjust_sequences = adjust_sequences
        self.s_df = s_df
        self.p_ids = torch.from_numpy(p_df.prompt_id.to_numpy())  # which is connection key of prompt & summaries
        self.p_titles = torch.from_numpy(p_df.prompt_title.to_numpy())
        self.p_texts = torch.from_numpy(p_df.prompt_text.to_numpy())
        self.p_questions = torch.from_numpy(p_df.prompt_question.to_numpy())
        self.s_ids = torch.from_numpy(s_df.prompt_id.to_numpy())  # which is connection key of prompt & summaries
        self.s_texts = torch.from_numpy(s_df.text.to_numpy())

    def __len__(self) -> int:
        return len(self.s_ids)

    def __getitem__(self, item: int) -> tuple[dict, Tensor]:
        # 1) load special token for making prompt sentence
        cls, sep = self.cfg.tokenizer.cls_token, self.cfg.tokenizer.sep_token
        com, tar = self.cfg.tokenizer.common_token, self.cfg.tokenizer.tar_token

        # 2) load feature for making prompt sentence & LLM's inputs
        key = find_index(self.p_ids, self.s_ids[item])

        # 3) make prompt sentence for LLM's inputs
        prompt = cls + tar + self.s_texts[item] + tar + sep
        prompt += com + self.p_questions[key] + com + self.p_titles[key] + com + self.p_texts[key] + com + sep

        inputs = self.tokenizing(self.cfg, prompt)
        labels = torch.as_tensor(self.s_df.iloc[item, 3:], dtype=torch.float)
        return inputs, labels


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
        p_df: prompt train dataset, prompts_train.csv
        s_df: prompt train dataset, summaries_train.csv
    """
    def __init__(self, cfg: configuration.CFG, p_df: pd.DataFrame, s_df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.p_df = p_df
        self.s_df = s_df
        self.tokenizing = tokenizing
        self.subsequent_tokenizing = subsequent_tokenizing
        self.adjust_sequences = adjust_sequences
        self.p_ids = torch.from_numpy(self.p_df.prompt_id.to_numpy())  # which is connection key of prompt & summaries
        self.p_titles = torch.from_numpy(self.p_df.prompt_title.to_numpy())
        self.p_texts = torch.from_numpy(self.p_df.prompt_text.to_numpy())
        self.p_questions = torch.from_numpy(self.p_df.prompt_question.to_numpy())
        self.s_ids = torch.from_numpy(self.s_df.prompt_id.to_numpy())  # which is connection key of prompt & summaries

    def __len__(self) -> int:
        return len(self.s_df)

    def __getitem__(self, item: int) -> dict:
        pass

