from utils import *
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
import os
import json
from typing import *
from tqdm import tqdm
import numpy as np
import argparse
import math
from tokenizer_13a import Tokenizer13a

bleu_calculator, rouge_calculator, meteor_calculator = None, None, None
bleu_tokenizer, sen_trans_tokenizer, sen_trans_model = None, None, None


def init_metrics(args):
    print("Init metrics...")
    global bleu_calculator, bleu_tokenizer, rouge_calculator, meteor_calculator, sen_trans_tokenizer, sen_trans_model
    bleu_calculator = evaluate.load(args.bleu_path)
    bleu_tokenizer = Tokenizer13a()
    rouge_calculator = evaluate.load(args.rouge_path)
    meteor_calculator = evaluate.load(args.meteor_path)
    sen_trans_tokenizer = AutoTokenizer.from_pretrained(args.sen_trans_path)
    sen_trans_model = AutoModel.from_pretrained(args.sen_trans_path).cuda()
    sen_trans_model.eval()
    print("Finish init metrics!")


def cal_at_k(args):
    k_list = args.k_list

    def at_k_list(one_data: Dict) -> Dict:
        summa_gt: str = one_data["ground_truth"]
        if args.language == "python":
            small_summa_gt = XLCoSTPythonSampleHelper.small_summa(summa_gt)
            big_summa_gt = XLCoSTPythonSampleHelper.big_summa(summa_gt)
        elif args.language == "java":
            small_summa_gt = XLCoSTJavaSampleHelper.small_summa(summa_gt)
            big_summa_gt = XLCoSTJavaSampleHelper.big_summa(summa_gt)

        max_bleu1, max_small_bleu1, max_big_bleu1 = 0.0, 0.0, 0.0
        max_bleu2, max_small_bleu2, max_big_bleu2 = 0.0, 0.0, 0.0
        max_bleu4, max_small_bleu4, max_big_bleu4 = 0.0, 0.0, 0.0
        max_big_rouge1, max_big_rouge2, max_big_rougeL, max_big_meteor = 0.0, 0.0, 0.0, 0.0
        max_small_rouge1, max_small_rouge2, max_small_rougeL, max_small_meteor = 0.0, 0.0, 0.0, 0.0
        max_big_sim, max_small_sim = 0.0, 0.0

        bleu1_list, small_bleu1_list, big_bleu1_list = [], [], []
        bleu2_list, small_bleu2_list, big_bleu2_list = [], [], []
        bleu4_list, small_bleu4_list, big_bleu4_list = [], [], []
        big_rouge1_list, big_rouge2_list, big_rougeL_list, big_meteor_list = [], [], [], []
        small_rouge1_list, small_rouge2_list, small_rougeL_list, small_meteor_list = [], [], [], []
        big_sim_list, small_sim_list = [], []

        N = min(len(one_data["all_gen_res"]), max(k_list))
        at_k_list_res = {}
        for i in range(N):
            gen_res = one_data["all_gen_res"][i]
          
            if args.language == "python":
                small_summa_res = XLCoSTPythonSampleHelper.small_summa(gen_res)
                big_summa_res = XLCoSTPythonSampleHelper.big_summa(gen_res)
            elif args.language == "java":
                small_summa_res = XLCoSTJavaSampleHelper.small_summa(gen_res)
                big_summa_res = XLCoSTJavaSampleHelper.big_summa(gen_res)
            small_summa_res = small_summa_res.strip()
            big_summa_res = big_summa_res.strip()

         
            bleu1, small_bleu1, big_bleu1 = 0.0, 0.0, 0.0
            bleu2, small_bleu2, big_bleu2 = 0.0, 0.0, 0.0
            bleu4, small_bleu4, big_bleu4 = 0.0, 0.0, 0.0
            big_rouge1, big_rouge2, big_rougeL, big_meteor = 0.0, 0.0, 0.0, 0.0
            small_rouge1, small_rouge2, small_rougeL, small_meteor = 0.0, 0.0, 0.0, 0.0
            big_similarity, small_similarity = 0.0, 0.0
            if len(gen_res.strip()) == 0:
                pass
            else:
                # bleu1 = get_bleu(prediction=gen_res, tokenizer=bleu_tokenizer,
                #                  reference=summa_gt, bleu=bleu_calculator, max_order=1)
                # bleu2 = get_bleu(prediction=gen_res, tokenizer=bleu_tokenizer,
                #                  reference=summa_gt, bleu=bleu_calculator, max_order=2)
                # bleu4 = get_bleu(prediction=gen_res, tokenizer=bleu_tokenizer,
                #                  reference=summa_gt, bleu=bleu_calculator, max_order=4)

                if args.need_small_bleu:
                    small_bleu1 = get_bleu(prediction=small_summa_res, tokenizer=bleu_tokenizer,
                                           reference=small_summa_gt, bleu=bleu_calculator, max_order=1, smooth=args.bleu_smooth)
                    small_bleu2 = get_bleu(prediction=small_summa_res, tokenizer=bleu_tokenizer,
                                           reference=small_summa_gt, bleu=bleu_calculator, max_order=2, smooth=args.bleu_smooth)
                    small_bleu4 = get_bleu(prediction=small_summa_res, tokenizer=bleu_tokenizer,
                                           reference=small_summa_gt, bleu=bleu_calculator, max_order=4, smooth=args.bleu_smooth)
                if args.need_big_bleu:
                    big_bleu1 = get_bleu(prediction=big_summa_res, tokenizer=bleu_tokenizer,
                                         reference=big_summa_gt, bleu=bleu_calculator, max_order=1, smooth=args.bleu_smooth)
                    big_bleu2 = get_bleu(prediction=big_summa_res, tokenizer=bleu_tokenizer,
                                         reference=big_summa_gt, bleu=bleu_calculator, max_order=2, smooth=args.bleu_smooth)
                    big_bleu4 = get_bleu(prediction=big_summa_res, tokenizer=bleu_tokenizer,
                                         reference=big_summa_gt, bleu=bleu_calculator, max_order=4, smooth=args.bleu_smooth)
                if args.need_small_rouge:
                    small_rouge1, small_rouge2, small_rougeL = get_rouge(prediction=small_summa_res, reference=small_summa_gt,
                                                                         rouge=rouge_calculator)
                if args.need_big_rouge:
                    big_rouge1, big_rouge2, big_rougeL = get_rouge(prediction=big_summa_res, reference=big_summa_gt,
                                                                   rouge=rouge_calculator)
                if args.need_small_meteor:
                    small_meteor = get_meteor(prediction=small_summa_res, reference=small_summa_gt,
                                              meteor=meteor_calculator)
                if args.need_big_meteor:
                    big_meteor = get_meteor(prediction=big_summa_res, reference=big_summa_gt,
                                            meteor=meteor_calculator)
                if args.need_small_sim:
                    small_similarity = get_similarity(prediction=small_summa_res, reference=small_summa_gt,
                                                      tokenizer=sen_trans_tokenizer, model=sen_trans_model)
                if args.need_big_sim:
                    big_similarity = get_similarity(prediction=big_summa_res, reference=big_summa_gt,
                                                    tokenizer=sen_trans_tokenizer, model=sen_trans_model)

            # bleu1_list.append(bleu1)
            # bleu2_list.append(bleu2)
            # bleu4_list.append(bleu4)
            if args.need_small_bleu:
                small_bleu1_list.append(small_bleu1)
                small_bleu2_list.append(small_bleu2)
                small_bleu4_list.append(small_bleu4)
            if args.need_big_bleu:
                big_bleu1_list.append(big_bleu1)
                big_bleu2_list.append(big_bleu2)
                big_bleu4_list.append(big_bleu4)
            if args.need_small_rouge:
                small_rouge1_list.append(small_rouge1)
                small_rouge2_list.append(small_rouge2)
                small_rougeL_list.append(small_rougeL)
            if args.need_big_rouge:
                big_rouge1_list.append(big_rouge1)
                big_rouge2_list.append(big_rouge2)
                big_rougeL_list.append(big_rougeL)
            if args.need_small_meteor:
                small_meteor_list.append(small_meteor)
            if args.need_big_meteor:
                big_meteor_list.append(big_meteor)
            if args.need_small_sim:
                small_sim_list.append(small_similarity)
            if args.need_big_sim:
                big_sim_list.append(big_similarity)

     
            # max_bleu1 = max(max_bleu1, bleu1)
            # max_bleu2 = max(max_bleu2, bleu2)
            # max_bleu4 = max(max_bleu4, bleu4)
            if args.need_small_bleu:
                max_small_bleu1 = max(max_small_bleu1, small_bleu1)
                max_small_bleu2 = max(max_small_bleu2, small_bleu2)
                max_small_bleu4 = max(max_small_bleu4, small_bleu4)
            if args.need_big_bleu:
                max_big_bleu1 = max(max_big_bleu1, big_bleu1)
                max_big_bleu2 = max(max_big_bleu2, big_bleu2)
                max_big_bleu4 = max(max_big_bleu4, big_bleu4)
            if args.need_small_rouge:
                max_small_rouge1 = max(max_small_rouge1, small_rouge1)
                max_small_rouge2 = max(max_small_rouge2, small_rouge2)
                max_small_rougeL = max(max_small_rougeL, small_rougeL)
            if args.need_big_rouge:
                max_big_rouge1 = max(max_big_rouge1, big_rouge1)
                max_big_rouge2 = max(max_big_rouge2, big_rouge2)
                max_big_rougeL = max(max_big_rougeL, big_rougeL)
            if args.need_small_meteor:
                max_small_meteor = max(max_small_meteor, small_meteor)
            if args.need_big_meteor:
                max_big_meteor = max(max_big_meteor, big_meteor)
            if args.need_small_sim:
                max_small_sim = max(max_small_sim, small_similarity)
            if args.need_big_sim:
                max_big_sim = max(max_big_sim, big_similarity)

            if i + 1 in k_list:
                # at_k_list_res[f"bleu-1@{i+1}"] = max_bleu1
                # at_k_list_res[f"bleu-2@{i+1}"] = max_bleu2
                # at_k_list_res[f"bleu-4@{i+1}"] = max_bleu4
                if args.need_small_bleu:
                    at_k_list_res[f"small_bleu-1@{i+1}"] = max_small_bleu1
                    at_k_list_res[f"small_bleu-2@{i+1}"] = max_small_bleu2
                    at_k_list_res[f"small_bleu-4@{i+1}"] = max_small_bleu4
                if args.need_big_bleu:
                    at_k_list_res[f"big_bleu-1@{i+1}"] = max_big_bleu1
                    at_k_list_res[f"big_bleu-2@{i+1}"] = max_big_bleu2
                    at_k_list_res[f"big_bleu-4@{i+1}"] = max_big_bleu4
                if args.need_small_rouge:
                    at_k_list_res[f"small_rouge-1@{i+1}"] = max_small_rouge1
                    at_k_list_res[f"small_rouge-2@{i+1}"] = max_small_rouge2
                    at_k_list_res[f"small_rouge-L@{i+1}"] = max_small_rougeL
                if args.need_big_rouge:
                    at_k_list_res[f"big_rouge-1@{i+1}"] = max_big_rouge1
                    at_k_list_res[f"big_rouge-2@{i+1}"] = max_big_rouge2
                    at_k_list_res[f"big_rouge-L@{i+1}"] = max_big_rougeL
                if args.need_small_meteor:
                    at_k_list_res[f"small_meteor@{i+1}"] = max_small_meteor
                if args.need_big_meteor:
                    at_k_list_res[f"big_meteor@{i+1}"] = max_big_meteor
                if args.need_small_sim:
                    at_k_list_res[f"small_sim@{i+1}"] = max_small_sim
                if args.need_big_sim:
                    at_k_list_res[f"big_sim@{i+1}"] = max_big_sim

        for i in k_list:
            # at_k_list_res[f"bleu-1_avg"] = np.mean(bleu1_list)
            # at_k_list_res[f"bleu-2_avg"] = np.mean(bleu2_list)
            # at_k_list_res[f"bleu-4_avg"] = np.mean(bleu4_list)
            if args.need_small_bleu:
                at_k_list_res[f"small_bleu-1_avg"] = np.mean(small_bleu1_list)
                at_k_list_res[f"small_bleu-2_avg"] = np.mean(small_bleu2_list)
                at_k_list_res[f"small_bleu-4_avg"] = np.mean(small_bleu4_list)
            if args.need_big_bleu:
                at_k_list_res[f"big_bleu-1_avg"] = np.mean(big_bleu1_list)
                at_k_list_res[f"big_bleu-2_avg"] = np.mean(big_bleu2_list)
                at_k_list_res[f"big_bleu-4_avg"] = np.mean(big_bleu4_list)
            if args.need_small_rouge:
                at_k_list_res[f"small_rouge-1_avg"] = np.mean(small_rouge1_list)
                at_k_list_res[f"small_rouge-2_avg"] = np.mean(small_rouge2_list)
                at_k_list_res[f"small_rouge-L_avg"] = np.mean(small_rougeL_list)
            if args.need_big_rouge:
                at_k_list_res[f"big_rouge-1_avg"] = np.mean(big_rouge1_list)
                at_k_list_res[f"big_rouge-2_avg"] = np.mean(big_rouge2_list)
                at_k_list_res[f"big_rouge-L_avg"] = np.mean(big_rougeL_list)
            if args.need_small_meteor:
                at_k_list_res[f"small_meteor_avg"] = np.mean(small_meteor_list)
            if args.need_big_meteor:
                at_k_list_res[f"big_meteor_avg"] = np.mean(big_meteor_list)
            if args.need_small_sim:
                at_k_list_res[f"small_sim_avg"] = np.mean(small_sim_list)
            if args.need_big_sim:
                at_k_list_res[f"big_sim_avg"] = np.mean(big_sim_list)
        return at_k_list_res

    def process_one_data(one_data: Dict):
        at_k_list_res = at_k_list(one_data)
        one_data.update(at_k_list_res)

    save_path = args.save_path
    print(f"current file: {save_path}")
    with open(save_path, 'r', encoding='utf-8') as f:
        experiment_result = json.load(f)
    all_problrm_res_data = experiment_result["allResult"] 
    for i, one_data in enumerate(tqdm(all_problrm_res_data)):
        print(f"******************data{i}--index{one_data['index']}*******************")
        process_one_data(one_data)

    if args.need_small_bleu:
        experiment_result[f"small_bleu-1_overall_avg"] = np.mean(
            [one_data[f"small_bleu-1_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"small_bleu-2_overall_avg"] = np.mean(
            [one_data[f"small_bleu-2_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"small_bleu-4_overall_avg"] = np.mean(
            [one_data[f"small_bleu-4_avg"] for one_data in all_problrm_res_data])
    if args.need_big_bleu:
        experiment_result[f"big_bleu-1_overall_avg"] = np.mean(
            [one_data[f"big_bleu-1_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"big_bleu-2_overall_avg"] = np.mean(
            [one_data[f"big_bleu-2_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"big_bleu-4_overall_avg"] = np.mean(
            [one_data[f"big_bleu-4_avg"] for one_data in all_problrm_res_data])
    if args.need_small_rouge:
        experiment_result[f"small_rouge-1_overall_avg"] = np.mean(
            [one_data[f"small_rouge-1_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"small_rouge-2_overall_avg"] = np.mean(
            [one_data[f"small_rouge-2_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"small_rouge-L_overall_avg"] = np.mean(
            [one_data[f"small_rouge-L_avg"] for one_data in all_problrm_res_data])
    if args.need_big_rouge:
        experiment_result[f"big_rouge-1_overall_avg"] = np.mean(
            [one_data[f"big_rouge-1_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"big_rouge-2_overall_avg"] = np.mean(
            [one_data[f"big_rouge-2_avg"] for one_data in all_problrm_res_data])
        experiment_result[f"big_rouge-L_overall_avg"] = np.mean(
            [one_data[f"big_rouge-L_avg"] for one_data in all_problrm_res_data])
    if args.need_small_meteor:
        experiment_result[f"small_meteor_overall_avg"] = np.mean(
            [one_data[f"small_meteor_avg"] for one_data in all_problrm_res_data])
    if args.need_big_meteor:
        experiment_result[f"big_meteor_overall_avg"] = np.mean(
            [one_data[f"big_meteor_avg"] for one_data in all_problrm_res_data])
    if args.need_small_sim:
        experiment_result[f"small_sim_overall_avg"] = np.mean(
            [one_data[f"small_sim_avg"] for one_data in all_problrm_res_data])
    if args.need_big_sim:
        experiment_result[f"big_sim_overall_avg"] = np.mean(
            [one_data[f"big_sim_avg"] for one_data in all_problrm_res_data])

    for i in k_list:
        # experiment_result[f"bleu-1@{i}_avg"] = np.mean(
        #     [one_data[f"bleu-1@{i}"] for one_data in all_problrm_res_data])
        # experiment_result[f"bleu-2@{i}_avg"] = np.mean(
        #     [one_data[f"bleu-2@{i}"] for one_data in all_problrm_res_data])
        # experiment_result[f"bleu-4@{i}_avg"] = np.mean(
        #     [one_data[f"bleu-4@{i}"] for one_data in all_problrm_res_data])
        if args.need_small_bleu:
            experiment_result[f"small_bleu-1@{i}_avg"] = np.mean(
                [one_data[f"small_bleu-1@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"small_bleu-2@{i}_avg"] = np.mean(
                [one_data[f"small_bleu-2@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"small_bleu-4@{i}_avg"] = np.mean(
                [one_data[f"small_bleu-4@{i}"] for one_data in all_problrm_res_data])
        if args.need_big_bleu:
            experiment_result[f"big_bleu-1@{i}_avg"] = np.mean(
                [one_data[f"big_bleu-1@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"big_bleu-2@{i}_avg"] = np.mean(
                [one_data[f"big_bleu-2@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"big_bleu-4@{i}_avg"] = np.mean(
                [one_data[f"big_bleu-4@{i}"] for one_data in all_problrm_res_data])
        if args.need_small_rouge:
            experiment_result[f"small_rouge-1@{i}_avg"] = np.mean(
                [one_data[f"small_rouge-1@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"small_rouge-2@{i}_avg"] = np.mean(
                [one_data[f"small_rouge-2@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"small_rouge-L@{i}_avg"] = np.mean(
                [one_data[f"small_rouge-L@{i}"] for one_data in all_problrm_res_data])
        if args.need_big_rouge:
            experiment_result[f"big_rouge-1@{i}_avg"] = np.mean(
                [one_data[f"big_rouge-1@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"big_rouge-2@{i}_avg"] = np.mean(
                [one_data[f"big_rouge-2@{i}"] for one_data in all_problrm_res_data])
            experiment_result[f"big_rouge-L@{i}_avg"] = np.mean(
                [one_data[f"big_rouge-L@{i}"] for one_data in all_problrm_res_data])
        if args.need_small_meteor:
            experiment_result[f"small_meteor@{i}_avg"] = np.mean(
                [one_data[f"small_meteor@{i}"] for one_data in all_problrm_res_data])
        if args.need_big_meteor:
            experiment_result[f"big_meteor@{i}_avg"] = np.mean(
                [one_data[f"big_meteor@{i}"] for one_data in all_problrm_res_data])
        if args.need_small_sim:
            experiment_result[f"small_sim@{i}_avg"] = np.mean(
                [one_data[f"small_sim@{i}"] for one_data in all_problrm_res_data])
        if args.need_big_sim:
            experiment_result[f"big_sim@{i}_avg"] = np.mean(
                [one_data[f"big_sim@{i}"] for one_data in all_problrm_res_data])
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="self-guided-cal-score")

    parser.add_argument('--k-list', nargs='+', type=int, required=True)
    parser.add_argument('--language', required=True, choices=['python', 'java'], type=str)
    parser.add_argument('--save-path', default=None, required=True, type=str) 

    parser.add_argument(
        '--bleu-path', default='/home/clw/hhk/CodeSecurity/Experiment/evaluate/metrics/bleu', type=str)
    parser.add_argument(
        '--rouge-path', default='/home/clw/hhk/CodeSecurity/Experiment/evaluate/metrics/rouge', type=str)
    parser.add_argument(
        '--meteor-path', default='/home/clw/hhk/CodeSecurity/Experiment/evaluate/metrics/meteor', type=str)
    
    parser.add_argument(
        '--sen-trans-path', default='/home/clw/hf_local_models/models--sentence-transformers--all-MiniLM-L6-v2', type=str)
    parser.add_argument('--bleu-smooth', action="store_true")
    parser.add_argument('--need-small-bleu', action="store_true") 
    parser.add_argument('--need-big-bleu', action="store_true")
    parser.add_argument('--need-small-rouge', action="store_true")
    parser.add_argument('--need-big-rouge', action="store_true")
    parser.add_argument('--need-small-meteor', action="store_true")
    parser.add_argument('--need-big-meteor', action="store_true")
    parser.add_argument('--need-small-sim', action="store_true")
    parser.add_argument('--need-big-sim', action="store_true")
    args = parser.parse_args()
    # print(args.k_list)
    print("*****args*****")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    init_metrics(args)
    cal_at_k(args)
