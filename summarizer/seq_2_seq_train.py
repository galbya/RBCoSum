from datetime import datetime
import json
import torch
from typing import *
import random
import numpy as np
import argparse
import transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os

from replace_data_fields import replace_rules
from seq_2_seq_dataset import Seq2SeqDataset
from seq_2_seq_data_collator import MyDataCollatorForSeq2Seq

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['RANK'] = '0'                    
os.environ['WORLD_SIZE'] = '1'               
os.environ['MASTER_ADDR'] = '127.0.0.1'   
os.environ['MASTER_PORT'] = '1234'         
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ["ACCELERATE_USE_DEEPSPEED"] = 'true'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)


def main(args):
    set_seed(args.seed)
    model_path = args.model_path

    #######################################
    print("Loading tokenizer...")
   
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    print("Finish loading tokenizer!")

    print("Loading dataset...")
    train_data, eval_data = get_dataset(args, tokenizer)
    print("Finish loading dataset!")

    print("Loading model ...")

    model = T5ForConditionalGeneration.from_pretrained(model_path)

    print("Finish loading model!")

    # Save command to file
    argsdict = vars(args)
    # print(argsdict)
    if not os.path.exists(args.save_dir_root):
        os.mkdir(args.save_dir_root)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    with open(os.path.join(args.save_dir, "command.json"), 'w', encoding='utf-8') as f:
        json.dump(argsdict, f, indent=2)

    run_training(args, train_data, eval_data, tokenizer, model)

    print("Saving model ...")
    model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))
    print("Finish save model!")


def get_dataset(args, tokenizer):
    train_new_fields_data = None
    train_replace_func = None
    if args.train_new_fields_data_path:
        train_replace_func = replace_rules[args.train_replace_rule]
        with open(args.train_new_fields_data_path, 'r', encoding='utf-8') as f:
            train_new_fields_data = json.load(f)

    eval_new_fields_data = None
    eval_replace_func = None
    if args.eval_new_fields_data_path:
        eval_replace_func = replace_rules[args.eval_replace_rule]
        with open(args.eval_new_fields_data_path, 'r', encoding='utf-8') as f:
            eval_new_fields_data = json.load(f)

    train_data = Seq2SeqDataset(data_path=args.apps_train_files, task_type=args.task_type, tokenizer=tokenizer,
                                max_encoder_tokens=args.max_encoder_tokens, max_decoder_tokens=args.max_decoder_tokens, language=args.language,
                                replace_fields_rule=train_replace_func, data_include_new_fields=train_new_fields_data)
    eval_data = Seq2SeqDataset(data_path=args.apps_eval_files, task_type=args.task_type, tokenizer=tokenizer,
                               max_encoder_tokens=args.max_encoder_tokens, max_decoder_tokens=args.max_decoder_tokens, language=args.language,
                               replace_fields_rule=eval_replace_func, data_include_new_fields=eval_new_fields_data)
    for i in range(3):
        print(eval_data.infer_input(i) + eval_data.ground_truth(i))
        print("********************")
    # exit(-1)
    return train_data, eval_data


def run_training(args, train_data: Seq2SeqDataset, eval_data: Seq2SeqDataset, tokenizer, model):
    start_iteration = 0

    ## Dataloading ########################################################
    train_data.start_iteration = start_iteration

    ## Start Loop ########################################################
    print("Setting up trainer ...")

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args.save_dir,

        # Use this to continue training if output_dir points to a checkpoint directory.
        overwrite_output_dir=False,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,  
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        # warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_steps=args.log_freq, 
        save_steps=args.save_freq,  
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        save_total_limit=1, 

        no_cuda=args.no_cuda,
        seed=args.seed,
        data_seed=args.seed,
        # local_rank=args.local_rank, 
        dataloader_drop_last=True,  
        dataloader_num_workers=2, 
        gradient_checkpointing=args.gradient_checkpointing, 
        # deepspeed=args.deepspeed,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    data_collator = MyDataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )
    print("Finish set up trainer!")

    print(f"Starting training...")
    ##########################
    # print(trainer.evaluate())
    trainer.train()
    print("Finish training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Modelling on Code")

    parser.add_argument('--model-path',
                        default=r"xxxx",
                        type=str)
    parser.add_argument('--task-type', default=None, required=True, type=str)
    parser.add_argument('--language', required=True, choices=['python', 'java'], type=str)

    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--no-cuda', default=False, type=bool)
    parser.add_argument('--gradient-checkpointing', action="store_true")
    # Dataloading
    parser.add_argument('--apps-dataroot', default='resources', type=str)
    parser.add_argument('--apps-train-files',
                        default='xxxx', type=str)
    parser.add_argument('--apps-eval-files',
                        default='xxxx', type=str)
    parser.add_argument('--train-replace-rule', default=None, type=str)
    parser.add_argument('--train-new-fields-data-path', default=None, type=str)  
    parser.add_argument('--eval-replace-rule', default=None, type=str)
    parser.add_argument('--eval-new-fields-data-path', default=None, type=str) 

    # Training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument(
        '--lr-scheduler', default="cosine", type=str)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--warmup-ratio', default=0.15, type=float)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--grad-acc-steps', default=2, type=int)

    parser.add_argument('--fp16', default=True)
    parser.add_argument('--bf16', default=False)

    parser.add_argument('--log-freq', default=290, type=int)
    parser.add_argument('--eval-steps', default=580, type=float)
    parser.add_argument('--save-freq', default=580, type=int)

    args = parser.parse_args()
    assert not (args.fp16 and args.bf16), "fp16 and bf16 cannot be True at the same time"
    # check train-replace-rule
    flag1, flag2 = False, False
    if args.train_replace_rule is not None:
        flag1 = True
    if args.train_new_fields_data_path is not None:
        flag2 = True
    assert not (flag1 ^ flag2)
    if flag1:
        assert args.train_replace_rule in replace_rules
    # check eval-replace-rule
    flag1, flag2 = False, False
    if args.eval_replace_rule is not None:
        flag1 = True
    if args.eval_new_fields_data_path is not None:
        flag2 = True
    assert not (flag1 ^ flag2)
    if flag1:
        assert args.eval_replace_rule in replace_rules

    if args.task_type == "gt_only_big":
        args.max_encoder_tokens = 448
        args.max_decoder_tokens = 32
    elif args.task_type == "in_small_gt_big":
        args.max_encoder_tokens = 680
        args.max_decoder_tokens = 32
    else:
        print("Wrong task_type!")
        exit(-1)

    args.save_dir_root = os.path.join("model_checkpoints", args.language)
    args.save_dir = os.path.join(
        args.save_dir_root, datetime.now().strftime("%m-%d-%H-%M"))
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    main(args)
