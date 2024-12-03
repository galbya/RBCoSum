from utils import *
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification,
                          BertForSequenceClassification, T5ForSequenceClassification)
import os
import json
from typing import *
import tqdm
import numpy as np
import argparse
import torch
import math

from decoder_only_dataset import DeOnlyDataset
from seq_2_seq_dataset import Seq2SeqDataset
from replace_data_fields import replace_rules

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MODEL = None
TOKENIZER = None


def model_load(args):
    global MODEL, TOKENIZER
    assert args.rrk_model_type in ["opt-350m", "codebert-base", "starencoder-125m",
                                   "codet5small", "codet5base", "codet5large"]
    print(f"Loading {args.rrk_model_type}...")

    TOKENIZER = AutoTokenizer.from_pretrained(args.rrk_model_path)

    if "opt" in args.rrk_model_type:
        MODEL = AutoModelForSequenceClassification.from_pretrained(args.rrk_model_path).cuda(args.device)
    elif "codebert" in args.rrk_model_type:
        MODEL = RobertaForSequenceClassification.from_pretrained(args.rrk_model_path).cuda(args.device)
    elif "starencoder" in args.rrk_model_type:
        MODEL = BertForSequenceClassification.from_pretrained(args.rrk_model_path).cuda(args.device)
    elif "codet5" in args.rrk_model_type:
        MODEL = T5ForSequenceClassification.from_pretrained(args.rrk_model_path).cuda(args.device)
    MODEL.eval()
    # print(MODEL)
    print(f"Finish loading {args.rrk_model_type}")


def model_batch_rerank(model_type, texts, device: str) -> List[float]:
    batch_rerank_scores = []
    if "opt" in model_type:
        batch_rerank_scores = opt_rerank(tokenizer=TOKENIZER, model=MODEL, texts=texts, device=device)
    elif "codebert" in model_type:
        batch_rerank_scores = codebert_rerank(tokenizer=TOKENIZER, model=MODEL, texts=texts, device=device)
    elif "starencoder" in model_type:
        batch_rerank_scores = starencoder_rerank(tokenizer=TOKENIZER, model=MODEL, texts=texts, device=device)
    elif "codet5" in model_type:
        batch_rerank_scores = codet5_rerank(tokenizer=TOKENIZER, model=MODEL, texts=texts, device=device)
    else:
        print(f"model type \"{model_type}\" error")
        exit(-1)
    return batch_rerank_scores


def get_prompts(model_type, all_gen_res: List[str], code) -> List[str]:
    if "opt" in model_type:
        prompt_template = "code: {code}query: {summary}"
        prompts = [prompt_template.format_map(
            {"code": code, "summary": summary.strip()}) for summary in all_gen_res]
    elif "codebert" in model_type:
        prompts = [summary.strip() + TOKENIZER.sep_token + code for summary in all_gen_res]
    elif "starencoder" in model_type:
        prompts = [TOKENIZER.cls_token + summary.strip() + TOKENIZER.sep_token +
                   code + TOKENIZER.sep_token for summary in all_gen_res]
    elif "codet5" in model_type:
        prompts = [summary.strip() + TOKENIZER.sep_token + code for summary in all_gen_res]
    return prompts


def process_one_data(data_index: int, args, summary_result: Dict):

    assert summary_result["allResult"][data_index]["index"] == data_index
    N = args.chosen_samples
    code = summary_result["allResult"][data_index]["code"]
    ground_truth = summary_result["allResult"][data_index]["ground_truth"]
    all_gen_res = summary_result["allResult"][data_index]["all_gen_res"]
    prompts = get_prompts(args.rrk_model_type, all_gen_res, code)
    prompts = prompts[:args.target_samples]
    # for p in prompts:
    #     print(p)
    # exit()

    bs = args.batch_per_data
    batches = [prompts[i:i + bs] for i in range(0, len(prompts), bs)]
    one_result = {"index": data_index,
                  "code": code,
                  "ground_truth": ground_truth,
                  "all_gen_res": []}
    scores = []
    for i, one_batch in enumerate(batches):
        print(f"data{data_index} rerank batch {i*bs}-{i*bs+len(one_batch)-1}")
        batch_rerank_scores = model_batch_rerank(model_type=args.rrk_model_type, texts=one_batch,
                                                 device=args.device)
        scores.extend(batch_rerank_scores)
    sorted_indices = np.argsort(-np.array(scores))
    top_n_indices = sorted_indices[:N]
    one_result["all_gen_res"].extend([all_gen_res[i] for i in top_n_indices])
    return one_result


def run_for_goal_data(args, goal_indexes: List, summary_result: Dict):
    save_path = args.save_path
    experiment_result = {}
    if os.path.exists(save_path): 
        with open(save_path, 'r', encoding='utf-8') as f:
            experiment_result = json.load(f)
    else:
        experiment_result = {"allResult": []}

    for index in goal_indexes:
        one_reslut = process_one_data(data_index=index, args=args, summary_result=summary_result)
        experiment_result["allResult"].append(one_reslut)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, indent=2, ensure_ascii=False)


def init_goal_indexes_and_model(args):
    
    with open(args.resource_path, 'r', encoding='utf-8') as f:
        summary_result = json.load(f)
    assert "allResult" in summary_result
    assert args.target_samples <= len(summary_result["allResult"][0]["all_gen_res"])

    if os.path.exists(args.save_path):
        with open(args.save_path, 'r', encoding='utf-8') as f:
            rrk_result = json.load(f)
        goal_indexes: List = []

        index_set = {d["index"] for d in rrk_result["allResult"]}
        for i in range(len(summary_result["allResult"])):
            if i not in index_set:
                goal_indexes.append(i)
    else:
        goal_indexes = [i for i in range(len(summary_result["allResult"]))]

    # for i in range(5):
    #     print(goal_ids[i])
    # exit(-1)

    print("result to be raranked:")
    for i, index in enumerate(goal_indexes):
        print(f"{i}--index {index}")

    model_load(args)
    return goal_indexes, summary_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="code sum inference")

    parser.add_argument('--rrk-model-type', default="", required=True, type=str)
    parser.add_argument('--rrk-model-path', default="", required=True, type=str)
    parser.add_argument('--language', required=True, choices=['python', 'java'], type=str)
    parser.add_argument('--origin-model-type', default="", required=True, type=str)  
    parser.add_argument('--chosen-samples', default=1, type=int)  
    parser.add_argument('--target-samples', required=True, type=int)  
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch-per-data', default=1, type=int)  
    parser.add_argument('--resource-path', default=None, required=True, type=str) 
    args = parser.parse_args()

    assert args.batch_per_data <= args.target_samples
    assert args.chosen_samples <= args.target_samples

    save_dir = rf"results/ranker/{args.rrk_model_type}/{args.language}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file_name = os.path.basename(args.resource_path)
    save_file_name_wo_suf = save_file_name[:save_file_name.rfind('.')]
    args.save_path = os.path.join(save_dir,
                                  f"{args.origin_model_type}_{save_file_name_wo_suf}_target{args.target_samples}_chosen{args.chosen_samples}.json")

    print("*****args*****")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    goal_indexes, summary_result = init_goal_indexes_and_model(args)
    run_for_goal_data(args, goal_indexes, summary_result)
