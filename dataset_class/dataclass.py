import gc
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, sampler, DataLoader

import configuration
from dataset_class.preprocessing import tokenizing, adjust_sequences
from dataset_class.preprocessing import subsequent_tokenizing, subsequent_decode, sequence_length, find_index
from trainer.trainer_utils import SmartBatchingSampler, SmartBatchingCollate


class OneToOneSmartBatchDataset(Dataset):
    """
    For Supervised Learning Pipeline, making "type 1" prompt sentence for LLMs Inputs
    This class have 3 functions
        1) apply preprocessing for mis-spelling words in prompt texts & summaries texts (maybe later applying)
        2) make prompt sentence with no padding (x)
        => this pipeline will be set hyper-params 'max_len' as at least 1920,
        which is maximum value of all instance's prompt sentence length
        3) get torch.utils.data.DataLoader instance with smart-batching
    Notes:
        This class has same function with OneToOneDataset, but this class apply for smart-batching
        (not map-style dataset, but iterable-style dataset)
    Args:
        cfg: config module from configuration.py
        p_df: prompt train dataset, prompts_train.csv
        s_df: prompt train dataset, summaries_train.csv
    """
    def __init__(self, cfg: configuration.CFG, s_df: pd.DataFrame) -> None:
        super(OneToOneSmartBatchDataset, self).__init__()
        self.cfg = cfg
        self.tokenizing = tokenizing
        self.s_df = s_df
        self.prompts = self.s_df.prompt_id.to_list()  # prompt sentence: already have shape of prompt
        self.labels = self.s_df.iloc[:, 3:5].to_list()  # native in original dataframe

    def __len__(self) -> int:
        return len(self.s_df)

    def __getitem__(self, item: int) -> tuple[dict, Tensor]:
        prompt = self.prompts[item]
        inputs = self.tokenizing(self.cfg, prompt, padding=False)
        labels = torch.as_tensor(self.labels[item], dtype=torch.float)

        del prompt
        gc.collect()
        return inputs, labels

    def get_smart_dataloader(self, drop_last: bool = True) -> DataLoader:
        """ function for getting smart-batching dataloader """
        collate_fn = SmartBatchingCollate(
            labels=self.labels,
            max_length=self.cfg.max_len,
            pad_token_id=self.cfg.tokenizer.pad_token_id
        )

        sampler = SmartBatchingSampler(
            data_instance=self.prompts,
            batch_size=self.cfg.batch_size,
        )

        smart_dataloader = DataLoader(
            dataset=self,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
        return smart_dataloader


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
        self.s_df = s_df
        self.p_ids = p_df.prompt_id.to_numpy()  # which is connection key of prompt & summaries
        self.p_titles = p_df.prompt_title.to_numpy()
        self.p_texts = p_df.prompt_text.to_numpy()
        self.p_questions = p_df.prompt_question.to_numpy()
        self.s_ids = s_df.prompt_id.to_numpy()  # which is connection key of prompt & summaries
        self.s_texts = s_df.text.to_numpy()

    def __len__(self) -> int:
        return len(self.s_ids)

    def __getitem__(self, item: int) -> tuple[dict, Tensor]:
        # 1) load special token for making prompt sentence
        cls, sep = self.cfg.tokenizer.cls_token, self.cfg.tokenizer.sep_token
        com, tar = self.cfg.tokenizer.common_token, self.cfg.tokenizer.tar_token

        # 2) load feature for making prompt sentence & LLM's inputs
        key = find_index(self.p_ids, self.s_ids[item])
        """
        3) make prompt sentence for LLM's inputs
            - summaries_text + prompt_question + prompt_title + prompt_text => 0
            - summaries_text + prompt_question + prompt_title
            - summaries_text + prompt_question + prompt_text
            - summaries_text + prompt_title + prompt_text
        """
        prompt = cls + tar + self.s_texts[item] + tar + sep
#        prompt += com + self.p_questions[key] + com + self.p_titles[key] + com + self.p_texts[key] + com + sep
        prompt += com + self.p_questions[key] + com + self.p_titles[key] + com + sep

        inputs = self.tokenizing(self.cfg, prompt)
        labels = torch.as_tensor(self.s_df.iloc[item, 3:5], dtype=torch.float)

        del prompt
        gc.collect()
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

