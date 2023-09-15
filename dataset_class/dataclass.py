import gc, ast, sys, random
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

import configuration
from dataset_class.preprocessing import tokenizing, adjust_sequences, subsequent_decode
from dataset_class.preprocessing import subsequent_tokenizing, subsequent_decode, sequence_length, find_index, cleaning_words
from trainer.trainer_utils import SmartBatchingSampler, SmartBatchingCollate


class OneToOneDataset(Dataset):
    """
    For Supervised Learning Pipeline, making "type 1" prompt sentence for LLMs Inputs
    This class have 2 functions
        1) apply preprocessing for mis-spelling words in prompt texts & summaries texts (applied)
        2) make prompt sentence with no padding (x)
        => this pipeline will be set hyper-params 'max_len' as at least, which is maximum value of all instance's prompt sentence length
        3) (+add) make masking tensor for p-tuning
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
        self.p_texts = p_df.prompt_text.to_numpy()  # caused mis-matching distribution of Validation & Test
        self.p_questions = p_df.prompt_question.to_numpy()
        self.s_ids = s_df.prompt_id.to_numpy()  # which is connection key of prompt & summaries
        self.s_texts = s_df.text.to_numpy()

    def __len__(self) -> int:
        return len(self.s_ids)

    def __getitem__(self, item: int) -> tuple[dict, Tensor]:
        # 1) load special token for making prompt sentence
        cls, sep = self.cfg.tokenizer.cls_token, self.cfg.tokenizer.sep_token
        anc, tar = self.cfg.tokenizer.anc_token, self.cfg.tokenizer.tar_token

        # 2) load feature for making prompt sentence & LLM's inputs
        key = find_index(self.p_ids, self.s_ids[item])
        """
        3) make prompt sentence for LLM's inputs laterly, check special token's position & numbers
        are affected to model's NLU performance
            - prompt_question + prompt_title + summaries_text
        """
        prompt = cls + anc + self.p_questions[key] + anc + self.p_titles[key] + anc + sep + tar + self.s_texts[item] + tar + sep
        inputs = self.tokenizing(self.cfg, prompt)
        labels = torch.tensor(self.s_df.iloc[item, 3:5], dtype=torch.float)
        return inputs, labels


class OneToOneSmartBatchDataset(Dataset):
    """
    For Supervised Learning Pipeline, making "type 1" prompt sentence for LLMs Inputs
    This class have 3 functions
        1) apply preprocessing for mis-spelling words in prompt texts & summaries texts (maybe later applying)
        2) make prompt sentence with no padding (x)
        => this pipeline will be set hyper-params 'max_len' as at least 1920,
        which is maximum value of all instance's prompt sentence length
        3) (+add) make masking tensor for p-tuning
        4) get torch.utils.data.DataLoader instance with smart-batching
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
        self.contents = self.s_df.content.to_list()
        self.wordings = self.s_df.wording.to_list()
        self._prompts = self.s_df.prompt.apply(self.cfg.tokenizer.tokenize).apply(self.cfg.tokenizer.convert_tokens_to_ids).to_list()
        self._labels = [[content, wording] for content, wording in zip(self.contents, self.wordings)]

    def __len__(self) -> int:
        return len(self.s_df)

    def __getitem__(self, item: int) -> Tuple[List, List]:
        inputs = self._prompts[item]
        labels = self._labels[item]
        return inputs, labels

    def get_smart_dataloader(self, drop_last: bool = True) -> DataLoader:
        """ function for getting smart-batching dataloader """
        collate_fn = SmartBatchingCollate(
            labels=self._labels,
            max_length=self.cfg.max_len,
            pad_token_id=self.cfg.tokenizer.pad_token_id
        )
        sampler = SmartBatchingSampler(
            data_instance=self._prompts,
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


class OneToManyDataset(Dataset):
    """
    For Supervised Learning Pipeline, making "type 2" prompt sentence for LLMs Inputs
    This class have 4 functions
        1) apply preprocessing for mis-spelling words in prompt texts & summaries texts
        2) split original data schema into several subsets
        3) make prompt sentence with no padding with random shuffle for each subset target instance
            => fixed_summaries_text, labels
    Args:
        cfg: config module from configuration.py
        s_df: OneToMany train dataframe, which is already applied
    """
    def __init__(self, cfg: configuration.CFG, s_df: pd.DataFrame, is_valid: bool = False) -> None:
        self.cfg = cfg
        self.s_df = s_df
        self.tokenizing = tokenizing
        self.subsequent_tokenizing = subsequent_tokenizing
        self.subsequent_decode = subsequent_decode
        self.adjust_sequences = adjust_sequences
        self.p_ids = self.s_df.prompt_id.to_numpy()
        self.p_questions = self.s_df.prompt_question.to_numpy()
        self.p_titles = self.s_df.prompt_title.to_numpy()
        self.s_texts = self.s_df.fixed_text.to_numpy()
        self.contents = self.s_df.content.to_numpy()
        self.wordings = self.s_df.wording.to_numpy()
        self._is_valid = is_valid

    def __len__(self) -> int:
        return len(self.p_ids)

    def __getitem__(self, item: int):
        content, wording = np.array(ast.literal_eval(self.contents[item])), np.array(ast.literal_eval(self.wordings[item]))
        text = np.array(ast.literal_eval(self.s_texts[item]))
        p_question, p_title = self.p_questions[item], self.p_titles[item]
        # Data Augmentation for train stage: random shuffle for target text in prompt
        if not self._is_valid:
            indices = list(range(len(content)))
            random.shuffle(indices)
            content = content[indices]
            wording = wording[indices]
            text = text[indices]

        # Apply text normalize, cleaning, no padding
        anchor_length = len(self.subsequent_tokenizing(self.cfg, cleaning_words(p_question))) + len(self.subsequent_tokenizing(self.cfg, cleaning_words(p_title)))
        fixed_length = len(text) + 5 + anchor_length
        tmp_token_list = []
        for idx in range(len(content)):
            tmp_token_list.append(self.subsequent_tokenizing(self.cfg, cleaning_words(text[idx])))
        adjust_inputs, _ = adjust_sequences(tmp_token_list, (self.cfg.max_len - fixed_length))
        for idx in range(len(content)):
            text[idx] = self.subsequent_decode(self.cfg, adjust_inputs[idx])

        # Make prompt sequence
        cls, sep = self.cfg.tokenizer.cls_token, self.cfg.tokenizer.sep_token
        anc, tar = self.cfg.tokenizer.anchor_token, self.cfg.tokenizer.tar_token
        prompts = cls + p_title + anc + p_question + anc + sep
        for targets in text:
            prompts += targets + tar
        prompts += sep
        inputs = self.tokenizing(self.cfg, prompts)

        # Make masking tensor
        # scores = np.array(list(zip(content, wording)))  # result of zipping content, wording array
        # target_mask = np.zeros(len([token for token in inputs['input_ids'] if token != 0]))  # not padding token
        # label_content = torch.full(
        #     [len([token for token in inputs['input_ids'] if token != 0])], -1, dtype=torch.float
        # )
        # label_wording = torch.full(
        #     [len([token for token in inputs['input_ids'] if token != 0])], -1, dtype=torch.float
        # )
        target_mask = np.zeros(len([token for token in inputs['input_ids']]))  # not padding token
        label_content = torch.full(
            [len([token for token in inputs['input_ids']])], -1, dtype=torch.float
        )
        label_wording = torch.full(
            [len([token for token in inputs['input_ids']])], -1, dtype=torch.float
        )
        cnt_anc, cnt_tar, cnt_sep, nth_target, prev_i = 0, 0, 0, -1, -1
        for i, input_id in enumerate(inputs['input_ids']):
            if input_id == self.cfg.tokenizer.tar_token_id:
                cnt_tar += 1
                if cnt_tar == len(text):
                    break
            if input_id == self.cfg.tokenizer.sep_token_id:
                cnt_sep += 1
            # count number of target tokens
            if cnt_sep == 1 and input_id not in [self.cfg.tokenizer.pad_token_id, self.cfg.tokenizer.sep_token_id,
                                                 self.cfg.tokenizer.tar_token_id, self.cfg.tokenizer.anchor_token_id]:
                if (i - prev_i) > 1:
                    nth_target += 1
                label_content[i], label_wording[i] = content[nth_target], wording[nth_target]
                target_mask[i] = 1
                prev_i = i

        return inputs, target_mask, label_content, label_wording
