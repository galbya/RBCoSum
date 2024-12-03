import torch
from typing import *
import json
from transformers import AutoTokenizer
import random


class DeOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, task_type: str, max_tokens: int, tokenizer: AutoTokenizer,
                 language: str, in_train=True,replace_fields_rule: Callable = None, data_include_new_fields=None):
        assert task_type in ["in_small_gt_big", "gt_only_big"]
        self.data_path = data_path
        self.task_type = task_type
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.language = language
        self.replace_fields_rule = replace_fields_rule
        self.data_include_new_fields = data_include_new_fields
        self.samples: List[Dict] = self._initialize()
        if in_train:
            random.shuffle(self.samples)

    def _initialize(self):
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
                "summarization": one_data["summarization"],  # sub + \n\t + big(python)
                "only_code": one_data["code_wo_docstring"]
            }
            all_samples.append(sample)
        if self.replace_fields_rule is not None:
            self.replace_fields_rule(all_samples, self.data_include_new_fields)

        print("Finish loading data!")
        # for i in range(3):
        #     print(all_samples[i])
        #     print("**************")
        return all_samples

    def __len__(self):
        return len(self.samples)

    def infer_input(self, idx):
        sample = self.samples[idx]
        if self.language == "python":
            if self.task_type == "in_small_gt_big":
                prompt = f'<fim_prefix>{sample["before_suffix"]}\t"""{sample["small_sum"]}\n\t<fim_suffix>"""\n{sample["after_suffix"]}<fim_middle>'
            else:
                prompt = f'<fim_prefix>{sample["before_suffix"]}\t"""<fim_suffix>"""\n{sample["after_suffix"]}<fim_middle>'
        elif self.language == "java":
            if self.task_type == "in_small_gt_big":
                prompt = f'<fim_prefix>{sample["before_suffix"]}/*{sample["small_sum"]}\n<fim_suffix>*/\n{sample["after_suffix"]}<fim_middle>'
            else:
                prompt = f'<fim_prefix>{sample["before_suffix"]}/*<fim_suffix>*/\n{sample["after_suffix"]}<fim_middle>'
        return prompt

    def only_code(self, idx):
        sample = self.samples[idx]
        return sample["only_code"]

    def ground_truth(self, idx):
        sample = self.samples[idx]
        if self.language == "python":
            if self.task_type == "gt_only_big" or self.task_type == "in_small_gt_big":
                return sample["big_sum"]
        elif self.language == "java":
            if self.task_type == "gt_only_big" or self.task_type == "in_small_gt_big":
                return sample["big_sum"]

    def __getitem__(self, idx):
        prompt = self.infer_input(idx)
        ans = self.ground_truth(idx)

        ans = ans + "<|endoftext|>"
        # print(prompt + ans)
        # prompt("***************************")
        # self.sample_lens.append(len(self.tokenizer.tokenize(prompt + ans)))

        prompt_tokens = self.tokenizer.tokenize(prompt)
        prompt_token_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        ans_tokens = self.tokenizer.tokenize(ans)
        ans_token_ids = self.tokenizer.convert_tokens_to_ids(ans_tokens)

        if len(prompt_token_ids) >= self.max_tokens:
            ans_token_ids = []
            prompt_token_ids = prompt_token_ids[:self.max_tokens]
        elif len(prompt_token_ids) + len(ans_token_ids) > self.max_tokens:
            input_len = len(prompt_token_ids) + len(ans_token_ids)
            surplus_len = input_len - self.max_tokens
            ans_token_ids = ans_token_ids[:len(ans_token_ids) - surplus_len]

        input_ids = (torch.LongTensor(prompt_token_ids),
                     torch.LongTensor(ans_token_ids)) 
        res = {"input_ids": input_ids}

        return res
