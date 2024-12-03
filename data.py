import math
import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer

from arguments import DataArguments


class DatasetForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            split: str = 'train',
    ):
        if split == 'train':
            if os.path.isdir(args.train_data):
                train_datasets = []
                for file in os.listdir(args.train_data):
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                         split='train')
                    train_datasets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(train_datasets)
            else:
                self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        elif split == 'valid':
            if os.path.isdir(args.valid_data):
                valid_datasets = []
                for file in os.listdir(args.valid_data):
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.valid_data, file),
                                                         split='train')
                    valid_datasets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(valid_datasets)
            else:
                self.dataset = datasets.load_dataset('json', data_files=args.valid_data, split='train')

        elif split == 'test':
            if os.path.isdir(args.test_data):
                test_datasets = []
                for file in os.listdir(args.test_data):
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.test_data, file))
                    test_datasets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(test_datasets)
            else:
                self.dataset = datasets.load_dataset('json', data_files=args.test_data)

        self.dataset = self.dataset.filter(lambda example: len(example['before_docstring']) > 0 and len(example['after_docstring']) > 0)
        self.dataset = self.dataset.select_columns(['before_docstring', 'after_docstring', 'gpt-summary', 'big_sum'])
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        before_docstring = self.dataset[item]['before_docstring']
        after_docstring = self.dataset[item]['after_docstring']
        sub_summary = self.dataset[item]['gpt-summary']
        big_summary = self.dataset[item]['big_sum']
        return before_docstring, after_docstring, sub_summary, big_summary


@dataclass
class PromptTuningCollator(DataCollatorWithPadding):
    """
    Data collator used for prompt tuning.
    """
    prompt_type: str = "FIM"
    fuse_sub_summary: bool = False

    def trunc_pad(self, input: List[int], max_length: int, pad_token_id: int, trunc_side: str = "right", padding_side: str = "right"):
        if len(input) > max_length:
            if trunc_side == "right":
                input = input[:max_length]
            elif trunc_side == "left":
                input = input[-max_length:]
            else:
                raise ValueError("Please specify a correct trunc_side, should be 'right' or 'left'.")
        else:
            if padding_side == "right":
                input = input + [pad_token_id] * (max_length - len(input))
            elif padding_side == "left":
                input = [pad_token_id] * (max_length - len(input)) + input
            else:
                raise ValueError("Please specify a correct padding_side, should be 'right' or 'left'.")
        return input

    def __call__(self, features):
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        batch_max_length = self.max_length
        max_len = 0
        for feat in features:
            # code, sub_summary, big_summary = feat[0], feat[1], feat[2]
            before_docstring, after_docstring, sub_summary, big_summary = feat[0], feat[1], feat[2], feat[3]
            if self.prompt_type == "FIM":

                left_prompt = before_docstring
                right_prompt = after_docstring
                prompt = f'<fim_prefix>{left_prompt}\t"""<fim_suffix>"""\n{right_prompt}<fim_middle>'
                if self.fuse_sub_summary:
                    # sub_summary = "\n\t".join(sub_summary.split(";")[0])
                    pattern = re.compile('(summary:)|(Summary:)')
                    sub_summary = pattern.sub('', sub_summary)
                    prompt = f'<fim_prefix>{left_prompt}"""{sub_summary}\n\t<fim_suffix>"""\n{right_prompt}<fim_middle>'

                answer = big_summary
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=batch_max_length)['input_ids']
                answer_ids = self.tokenizer(answer, add_special_tokens=False, truncation=True, max_length=batch_max_length)['input_ids']

                input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]
                attention_masks = [1.0] * len(input_ids)
                labels = [-100] * len(prompt_ids) + answer_ids + [self.tokenizer.eos_token_id]

                assert len(input_ids) == len(attention_masks) == len(labels)
                max_len = max(max_len, len(input_ids))

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_masks)
                batch_labels.append(labels)

        batch_max_length = min(batch_max_length, max_len)
        # trunc and pad
        batch_input_ids = [self.trunc_pad(input_ids, batch_max_length, self.tokenizer.pad_token_id) for input_ids in batch_input_ids]
        batch_attention_masks = [self.trunc_pad(attention_masks, batch_max_length, 0) for attention_masks in batch_attention_masks]
        batch_labels = [self.trunc_pad(labels, batch_max_length, -100) for labels in batch_labels]

        batch = {
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_masks),
            "labels": torch.tensor(batch_labels),
        }
        return batch
