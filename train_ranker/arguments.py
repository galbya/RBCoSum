import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from datetime import datetime


@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    eval_data: Optional[str] = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class ScriptTrainingArguments(TrainingArguments): 
    output_dir: str = field(
        default="outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "random seed for initialization"},
    )
    activation_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to use activation checkpointing."},
    )
    deepspeed_plugin: Optional[str] = field(
        default=None,
        metadata={"help": "The deepspeed plugin to use."},
    )
    def __post_init__(self):
        super().__post_init__()

        task_name = f"xlcost_rerank_time_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.output_dir = os.path.join(self.output_dir, task_name)
        self.logging_dir = os.path.join(self.output_dir, "logs")
