from dataclasses import dataclass
import torch
from typing import *
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import torch.nn.functional as F


@dataclass
class MyDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # print(f"features: {features}")
        max_len = -1
        for i, input_ids_dict in enumerate(features):
            prompt_token_ids, ans_token_ids = input_ids_dict["input_ids"]
            input_token_ids = torch.cat((prompt_token_ids, ans_token_ids))
            labels_mask = torch.zeros(input_token_ids.size(0),
                                      dtype=torch.long)
            labels_mask[len(prompt_token_ids):] = 1


            labels = input_token_ids.clone().detach()
            labels[labels_mask == 0] = -100

            features[i]["input_ids"] = input_token_ids
            max_len = max(max_len, input_token_ids.size(0))
            features[i]["labels"] = labels

        for i, item in enumerate(features):
            label_len = item["labels"].size(0)
            if label_len < max_len:
                features[i]["labels"] = torch.cat((item["labels"],
                                                   torch.full((max_len - label_len,), -100)))

        # print(f"new features: ")
        # for f in features:
        #     for k, v in f.items():
        #         print(f"*** {k} ***")
        #         print(v)
        #         print(v.shape)
        #     print("############")

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # print(batch["input_ids"].shape)
        return batch
