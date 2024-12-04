import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import *
from transformers import (AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel,
                          T5ForConditionalGeneration)

import json
from typing import *
import tqdm
import numpy as np
import argparse
# from peft import get_peft_config, get_peft_model, TaskType, LoraConfig
# from peft import PeftModel, PeftConfig
import torch
import random

from decoder_only_dataset import DeOnlyDataset
from seq_2_seq_dataset import Seq2SeqDataset
from replace_data_fields import replace_rules


MODEL = None
TOKENIZER = None
TYPE_2_IDS = None
SPECIAL_TOKENS = None

random.seed(42)


def model_load(args):
    global MODEL, TOKENIZER, TYPE_2_IDS, SPECIAL_TOKENS
    assert args.model_type in ["codet5large", "starcoder1b"]
    print(f"Loading {args.model_type}...")

    if "codet5" in args.model_type:
        TOKENIZER = AutoTokenizer.from_pretrained(args.model_path)
    else:
        TOKENIZER = AutoTokenizer.from_pretrained(args.model_path)
        TOKENIZER.padding_side = 'left'
        TOKENIZER.pad_token = TOKENIZER.eos_token

    if "starcoder" in args.model_type:
        MODEL = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).cuda(args.device)
    elif "codet5" in args.model_type:
        MODEL = T5ForConditionalGeneration.from_pretrained(args.model_path,
                                                           torch_dtype=torch.float16).cuda(args.device)
    MODEL.eval()
    print(f"Finish loading {args.model_type}")


def model_batch_gen(model_type, texts, max_to_generate, do_sample, top_p, temperature,
                    use_beam_search, beam_size, device: str) -> List[str]:
    batch_gen_res = []
    if model_type == "starcoder1b":
        batch_gen_res = starcoder_generate(tokenizer=TOKENIZER, model=MODEL, texts=texts, use_beam_search=use_beam_search, beam_size=beam_size,
                                           max_to_generate=max_to_generate, do_sample=do_sample, top_p=top_p, temperature=temperature, device=device)
    elif "codet5" in model_type:
        batch_gen_res = codet5_generate(tokenizer=TOKENIZER, model=MODEL, texts=texts, use_beam_search=use_beam_search, beam_size=beam_size,
                                        max_to_generate=max_to_generate, do_sample=do_sample, top_p=top_p, temperature=temperature, device=device)
    else:
        print(f"model type \"{model_type}\" error")
        exit(-1)
    return batch_gen_res


def process_one_data(data_index: int, data_set: Union[DeOnlyDataset, Seq2SeqDataset], args):
    N = args.num_samples
    if args.wCCOT_rate is not None:
        prompt = data_set.infer_input_under_wCCOT(idx=data_index, wccot_rate=args.wCCOT_rate,
                                                  wccot_type=args.wCCOT_type)
    else:
        prompt = data_set.infer_input(idx=data_index)
    # print(prompt)
    prompts = [prompt for _ in range(N)]
    bs = args.batch_per_data
    batches = [prompts[i:i + bs] for i in range(0, N, bs)]
    one_result = {"index": data_index,
                  "code": data_set.only_code(data_index),
                  "ground_truth": data_set.ground_truth(data_index),
                  "all_gen_res": []}
    for i, one_batch in enumerate(batches):
        print(f"data{data_index} genetate batch {i*bs}-{i*bs+len(one_batch)-1}")
        batch_gen_res = model_batch_gen(model_type=args.model_type, texts=one_batch, max_to_generate=args.max_to_generate,
                                        do_sample=args.do_sample, top_p=0.95, temperature=args.temperature, device=args.device,
                                        use_beam_search=args.use_beam_search, beam_size=args.beam_size)
        one_result["all_gen_res"].extend(batch_gen_res)
    return one_result


def run_for_goal_data(args, goal_indexes: List, data_set: DeOnlyDataset):

    save_path = args.save_path
    experiment_result = {}
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            experiment_result = json.load(f)
    else:
        experiment_result = {"allResult": []}

    for index in goal_indexes:
        one_reslut = process_one_data(data_index=index, data_set=data_set, args=args)
        experiment_result["allResult"].append(one_reslut)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, indent=2, ensure_ascii=False)


def init_goal_indexes_and_model(args):
    new_fields_data = None
    replace_func = None
    if args.new_fields_data_path:
        replace_func = replace_rules[args.replace_rule]
        with open(args.new_fields_data_path, 'r', encoding='utf-8') as f:
            new_fields_data = json.load(f)

    if "codet5" in args.model_type:
        test_set = Seq2SeqDataset(data_path=args.resource_path, task_type=args.task_type, tokenizer=None, language=args.language,
                                  in_train=False, max_encoder_tokens=-1, max_decoder_tokens=-1,
                                  replace_fields_rule=replace_func, data_include_new_fields=new_fields_data)
    else:
        test_set = DeOnlyDataset(args.resource_path, task_type=args.task_type, max_tokens=-1, tokenizer=None, language=args.language,
                                 in_train=False, replace_fields_rule=replace_func, data_include_new_fields=new_fields_data)
    for i in range(3):
        if args.wCCOT_rate is not None:
            print(test_set.infer_input_under_wCCOT(i, args.wCCOT_rate, args.wCCOT_type) + test_set.ground_truth(i))
            print("********************")
        else:
            print(test_set.infer_input(i) + test_set.ground_truth(i))
            print("********************")

    if os.path.exists(args.save_path):
        with open(args.save_path, 'r', encoding='utf-8') as f:
            experiment_result = json.load(f)
        goal_indexes: List = []

        index_set = {d["index"] for d in experiment_result["allResult"]}
        for i in range(len(test_set)):
            if i not in index_set:
                goal_indexes.append(i)
    else:
        goal_indexes = [i for i in range(len(test_set))]

    # for i in range(5):
    #     print(goal_ids[i])
    # exit(-1)

    print("problem to be generate:")
    for i, index in enumerate(goal_indexes):
        print(f"{i}--index {index}")

    model_load(args)
    return goal_indexes, test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="code sum inference")

    parser.add_argument('--model-type', default="", required=True, type=str)
    parser.add_argument('--model-path', default="", required=True, type=str)
    parser.add_argument('--language', required=True, choices=['python', 'java'], type=str)
    parser.add_argument('--num-samples', default=1, type=int) 
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch-per-data', default=1,
                        type=int)  
    parser.add_argument('--use-beam-search', action="store_true")  
    parser.add_argument('--do-sample', action="store_true")  
    parser.add_argument('--beam-size', default=-1, type=int)

    parser.add_argument('--max-to-generate', default=32, type=int)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--task-type', default=None, required=True, type=str)
    parser.add_argument('--test-dataset-type', default="xlcost", required=True, type=str)  
    parser.add_argument('--train-dataset-type', default="xlcost", required=True, type=str) 
    parser.add_argument('--resource-path', default=None, required=True, type=str)
    parser.add_argument('--replace-rule', default=None, type=str)
    parser.add_argument('--new-fields-data-path', default=None, type=str)

    parser.add_argument('--wCCOT-rate', default=None, choices=[0.25, 0.5, 0.75, 1.0], type=float)
    parser.add_argument('--wCCOT-type', default=None, choices=["random", "replace"], type=str)

    args = parser.parse_args()

    assert args.batch_per_data <= args.num_samples
    flag1, flag2 = False, False
    if args.replace_rule is not None:
        flag1 = True
    if args.new_fields_data_path is not None:
        flag2 = True
    assert not (flag1 ^ flag2)
    if flag1:
        assert args.replace_rule in replace_rules

    if args.use_beam_search:
        assert args.beam_size >= 1
        assert args.do_sample == False
        args.temperature = None

    if args.do_sample == False:
        assert args.num_samples == 1, "There is no need to set num_samples > 1 without sampling!"
        args.temperature = None

    if args.wCCOT_rate is not None or args.wCCOT_type is not None:
        assert args.task_type == "in_small_gt_big", "Only 'in_small_gt_big' has CCOT!"
    if args.wCCOT_type is not None:
        assert args.wCCOT_rate is not None, "'wCCOT_type' needs to have a corresponding 'wCCOT_rate'!"

    save_dir = rf"results/{args.model_type}/{args.language}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sample_str = "_nosam" if args.do_sample == False else ""
    wccot_str = f'_wccot{args.wCCOT_rate if args.wCCOT_rate is not None else ""}{args.wCCOT_type if args.wCCOT_type is not None else ""}'
    if wccot_str == "_wccot":
        wccot_str = ""
    args.save_path = os.path.join(save_dir,
                                  f"ft_train_{args.train_dataset_type}_test_{args.test_dataset_type}_{args.task_type}{sample_str}{wccot_str}_result900_{args.num_samples}samples.json")

    print("*****args*****")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    goal_indexes, test_set = init_goal_indexes_and_model(args)
    run_for_goal_data(args, goal_indexes, data_set=test_set)
