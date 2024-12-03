from transformers.utils import PaddingStrategy
import torch
from typing import *
import json
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import random
from tqdm import tqdm
import math

from seq_2_seq_data_collator import MyDataCollatorForSeq2Seq


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, task_type: str, max_encoder_tokens: int, max_decoder_tokens: int,
                 tokenizer: AutoTokenizer, language: str,
                 in_train=True, replace_fields_rule: Callable = None, data_include_new_fields=None):
        assert task_type in ["in_small_gt_big", "gt_only_big"]
        self.data_path = data_path
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.language = language
        if language == "python":
            self.prefix = "summarize python: "
        elif language == "java":
            self.prefix = "summarize java: "
        self.max_encoder_tokens = max_encoder_tokens
        self.max_decoder_tokens = max_decoder_tokens
        if self.max_encoder_tokens < 0:
            self.max_encoder_tokens = 1024
        if self.max_decoder_tokens < 0:
            self.max_decoder_tokens = 512
        self.in_train = in_train
        self.replace_fields_rule = replace_fields_rule
        self.data_include_new_fields = data_include_new_fields
        self.samples: List[Dict] = self._initialize_samples()
        if tokenizer is not None:
            self.tokenized_samples: List[Dict] = self._initialize_tokenized_samples()
        else:
            print("[WARNING] tokenizer is None in Seq2SeqDataset. Ignore tokenizing samples.")

        self.subsum_steps: List[List[str]] = []

    def _initialize_samples(self):
        all_samples: List[Dict] = []
        print(f"Loading data {self.data_path} ...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        for one_data in datas:
            sample = {
                "before_suffix": one_data["before_docstring"],
                "after_suffix": one_data["after_docstring"],
                "small_sum": one_data["sub_sum"],
                "big_sum": one_data["big_sum"],
                "summarization": one_data["summarization"],  # sub + \n\t + big
                "only_code": one_data["code_wo_docstring"]
            }
            all_samples.append(sample)
        if self.replace_fields_rule is not None:
            self.replace_fields_rule(all_samples, self.data_include_new_fields)
        if self.in_train:
            random.shuffle(all_samples)
        return all_samples

    def _initialize_tokenized_samples(self):
        tokenized_samples: List[Dict] = []
        sources_len = []
        tgt_len = []
        for index, _ in enumerate(tqdm(self.samples)):
            source_ids = self.infer_input_ids_with_bos_eos(index)
            tgt_ids = self.ground_truth_ids_with_bos_and_eos(index)
            sources_len.append(len(source_ids))
            tgt_len.append(len(tgt_ids))
            sample = {"source_ids": source_ids, "tgt_ids": tgt_ids}
            tokenized_samples.append(sample)

        assert len(self.samples) == len(tokenized_samples)
        print("Finish loading data!")
        return tokenized_samples

    def __len__(self):
        return len(self.samples)

    def infer_input(self, idx: int):
        sample = self.samples[idx]
        if self.language == "python":
            if self.task_type == "in_small_gt_big":
                prompt = f'{self.prefix}{sample["before_suffix"]}\t"""{sample["small_sum"]}"""\n{sample["after_suffix"]}'
            else:
                prompt = f'{self.prefix}{self.only_code(idx)}'
        elif self.language == "java":
            if self.task_type == "in_small_gt_big":
                prompt = f'{self.prefix}{sample["before_suffix"]}/*{sample["small_sum"]}*/\n{sample["after_suffix"]}'
            else:
                prompt = f'{self.prefix}{self.only_code(idx)}'
        return prompt

    def infer_input_under_wCCOT(self, idx: int, wccot_rate: float, wccot_type: str):
        assert self.in_train == False
        sample = self.samples[idx]
        subsum_steps: List[str] = sample["small_sum"].split(" ; ")
        
        used_number = math.ceil(len(subsum_steps) * wccot_rate)
        if wccot_type is None:
            chosen_steps = subsum_steps[:used_number]
            subsum = " ; ".join(chosen_steps)
        else:
            if wccot_type == "random":
                shuffled_part = subsum_steps[:used_number]
                random.shuffle(shuffled_part)
                subsum_steps[:used_number] = shuffled_part
                subsum = " ; ".join(subsum_steps)
            elif wccot_type == "replace": 
                if len(self.subsum_steps) == 0:
                    for i, s in enumerate(self.samples):
                        self.subsum_steps.append(s["small_sum"].split(" ; "))
                
                other_steps = []
                for i, steps in enumerate(self.subsum_steps):
                    if i != idx:
                        other_steps.extend(steps)
                chosen_steps = random.sample(other_steps, used_number)
                subsum_steps[:used_number] = chosen_steps
                subsum = " ; ".join(subsum_steps)
        prompt = f'{self.prefix}{sample["before_suffix"]}\t"""{subsum}"""\n{sample["after_suffix"]}'
        return prompt

    def infer_input_ids_with_bos_eos(self, idx: int):
        prompt = self.infer_input(idx)
        prompt_tokens = self.tokenizer.tokenize(prompt)
        prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        prompt_ids = [self.tokenizer.bos_token_id] + \
            prompt_ids[:self.max_encoder_tokens - 2] + [self.tokenizer.eos_token_id]
        return prompt_ids

    def only_code(self, idx: int):
        sample = self.samples[idx]
        code = f'{sample["before_suffix"]}{sample["after_suffix"]}'
        return code

    def ground_truth(self, idx: int):
        
        sample = self.samples[idx]
        if self.task_type == "gt_only_big" or self.task_type == "in_small_gt_big":
            return sample["big_sum"]
        else:
            print("Invalid task type!")
            exit(-1)

    def ground_truth_ids_with_bos_and_eos(self, idx):
        gt = self.ground_truth(idx)
        gt_tokens = self.tokenizer.tokenize(gt)
        gt_ids = self.tokenizer.convert_tokens_to_ids(gt_tokens)
        gt_ids = [self.tokenizer.bos_token_id] + \
            gt_ids[:self.max_decoder_tokens - 2] + [self.tokenizer.eos_token_id]
        return gt_ids

    def __getitem__(self, idx: int):
        source_ids = self.tokenized_samples[idx]["source_ids"]
        tgt_ids = self.tokenized_samples[idx]["tgt_ids"]

        res = {"input_ids": source_ids, "labels": tgt_ids}
        return res

