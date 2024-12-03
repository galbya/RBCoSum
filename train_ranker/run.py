import logging
import os
from pathlib import Path
import json
from dataclasses import asdict
from transformers import AutoConfig, AutoTokenizer, RobertaTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from datetime import datetime

from arguments import ModelArguments, DataArguments, ScriptTrainingArguments
from data import DatasetForCE, GroupCollator
from modeling import CrossEncoder
from trainer import CETrainer

logger = logging.getLogger(__name__)


def save_args(args: dict, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    script_name = os.path.basename(__file__).split(".")[0]
    output_args_file = os.path.join(output_dir, f"{script_name}_args.json")
    with open(output_args_file, "w") as f:
        json.dump(args, f, indent=2)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ScriptTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    args = {"model_args": asdict(model_args), "data_args": asdict(
        data_args), "training_args": asdict(training_args)}
    save_args(args, training_args.output_dir)
    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    _model_class = CrossEncoder

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    if training_args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = DatasetForCE(data_args, tokenizer=tokenizer, mode='train')
    logger.info("***** Dataset info *****\n")
    logger.info("Number of samples: %d", len(train_dataset))
    logger.info("Number of samples with calling __getitem__: %d",
                len(train_dataset[0]))
    logger.info(tokenizer.decode(train_dataset[0][0]['input_ids']))

    eval_dataset = DatasetForCE(data_args, tokenizer=tokenizer, mode='eval')

    _trainer_class = CETrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()


if __name__ == "__main__":
    main()
