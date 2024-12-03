from typing import *
import random
import torch
import torch.nn.functional as F
from tokenizer_13a import Tokenizer13a
import re


class XLCoSTPythonSampleHelper:
    def code_str(origin_code: str):
        code_token_list = origin_code.strip().split(' ')
        indent_cnt = 0
        i = 0
        res = ""
        while i < len(code_token_list):
            new_token = ""
            if code_token_list[i] == "NEW_LINE":
                new_token = '\n'
            elif code_token_list[i] == "INDENT":
                indent_cnt += 1
                new_token = '\t' * indent_cnt
            elif code_token_list[i] == "DEDENT":
                indent_cnt -= 1
                i += 1
               
                while i < len(code_token_list) and code_token_list[i] == "DEDENT":
                    indent_cnt -= 1
                    i += 1
                new_token = '\t' * indent_cnt if indent_cnt > 0 else ""
                i -= 1  
            else:  
                if code_token_list[i - 1] == "NEW_LINE":
                    new_token = '\t' * indent_cnt + code_token_list[i] + \
                        ' ' if indent_cnt > 0 else code_token_list[i] + ' '
                else:
                    new_token = code_token_list[i] + ' '
            # print(i, code_token_list[i], [new_token])
            i += 1

            res += new_token

        res = res.strip(' ')
        res = res.replace(" _ _ main _ _ ", "__main__")
        func_begin = res.find("def")
        func_end = func_begin
        while (res[func_end] != '\n'):
            func_end += 1
        before_suffix = res[:func_end + 1]
        after_suffix = res[func_end + 1:]
        return before_suffix, after_suffix

    def summarization(origin_sum_text: str):
        
        index = origin_sum_text.rfind('|')
        over_all_sum = origin_sum_text[:index]
        step_sums = origin_sum_text[index + 1:]
        return step_sums.strip(' ') + "\n\t" + over_all_sum.strip(' ')

    def small_summa(summa_str: str):
        index = summa_str.rfind("\n\t")
        if index < 0:
            index = summa_str.find('\n')
        return summa_str[:index] if index > 0 else summa_str

    def big_summa(summa_str: str):
        index = summa_str.rfind("\n\t")
        offset = 2
        if index < 0:
            index = summa_str.find('\n')
            offset = 1
        return summa_str[index + offset:] if index > 0 else summa_str


class XLCoSTJavaSampleHelper:
    def code_str(origin_code: str):
        class_begin = origin_code.find("public class")
        if class_begin < 0:
            class_begin = origin_code.find("class")
        before_suffix = origin_code[:class_begin]
        before_suffix = before_suffix.strip() + '\n'

        after_suffix = origin_code[class_begin:].strip() + '\n'
        return before_suffix, after_suffix

    def summarization(origin_sum_text: str):
        index = origin_sum_text.rfind('|')
        over_all_sum = origin_sum_text[:index]
        step_sums = origin_sum_text[index + 1:]
        return step_sums.strip(' ') + "\n" + over_all_sum.strip(' ')

    def small_summa(summa_str: str):
        index = summa_str.rfind("\n\t")
        if index < 0:
            index = summa_str.find('\n')
        return summa_str[:index] if index > 0 else summa_str

    def big_summa(summa_str: str):
        index = summa_str.rfind("\n\t")
        offset = 2
        if index < 0:
            index = summa_str.find('\n')
            offset = 1
        return summa_str[index + offset:] if index > 0 else summa_str


class CodeSumSampleHelper:
    def code_str(origin_code: str):
        def remove_docstring(code):
            pattern = r'(r|)(["\']{3})(.*?)(\2)'
            return re.sub(pattern, '', code, count=1, flags=re.DOTALL)
        before_suffix_end = origin_code.find('r"""')
        if before_suffix_end < 0:
            before_suffix_end = origin_code.find("r'''")
        if before_suffix_end < 0:
            before_suffix_end = origin_code.find('"""')
        if before_suffix_end < 0:
            before_suffix_end = origin_code.find("'''")
        assert before_suffix_end > 0
        code_wo_docstring = remove_docstring(origin_code)
        before_suffix = code_wo_docstring[:before_suffix_end].rstrip(' ').rstrip('\t')
        after_suffix = code_wo_docstring[before_suffix_end:]
        begin_index = 0
        while begin_index < len(after_suffix) and after_suffix[begin_index] != '\n':
            if after_suffix[begin_index] == ' ' or after_suffix[begin_index] == '\t':
                begin_index += 1
                continue
            else:
                begin_index = -1
                break
        if begin_index == len(after_suffix):
            begin_index = -1
        after_suffix = after_suffix[begin_index + 1:] + '\n'
        return before_suffix, after_suffix

    def summarization(small_summa: str, big_summa: str):
        return small_summa.strip(' ') + "\n\t" + big_summa.strip(' ')


def tokenize_batch(batch_text: List[str], tokenizer, padding=False, truncation=False, max_length=None) -> List[List[str]]:
    tokenized_batch = []
    max_text_len = -1
    for text in batch_text:
        tokens = []
        if max_length is None:
            tokens = tokenizer.tokenize(text)
        else:
            assert truncation == True, "you should use `truncation=True` with `max_length`"
            tokens = tokenizer.tokenize(text, max_length=max_length, truncation=truncation)
        tokenized_batch.append(tokens)
        max_text_len = max(max_text_len, len(tokens))
    if padding == True:
        for i, tokens in enumerate(tokenized_batch):
            pad_len = max_text_len - len(tokens)
            if tokenizer.padding_side == "left":
                tokenized_batch[i] = [tokenizer.pad_token] * pad_len + tokenized_batch[i]
            else:
                tokenized_batch[i] = tokenized_batch[i] + [tokenizer.pad_token] * pad_len
    return tokenized_batch


def starcoder_generate(tokenizer, model, texts, max_to_generate, do_sample=None, top_p=None, temperature=None,
                       use_beam_search: bool = None, beam_size: int = None, device: str = None) -> List[str]:
    def starcoder_family(inputs, max_length, top_p, temperature):
        with torch.no_grad():
            if use_beam_search:
                assert len(texts) == 1, "There is no need to use beam search with batch size > 1"
                samples = model.generate(**inputs, max_length=max_length, num_beams=beam_size,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
            else:
                if do_sample == False:  
                    samples = model.generate(**inputs, max_length=max_length, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
                else:  
                    samples = model.generate(**inputs, max_length=max_length, top_p=top_p,
                                             temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        trunked_len = inputs['input_ids'].shape[1]
        samples = samples[:, trunked_len:] 
        result = tokenizer.batch_decode(samples, skip_special_tokens=True)
        return result

    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048)
    inputs["input_ids"] = inputs["input_ids"].cuda(device)
    # print(inputs["input_ids"].shape)
    inputs["attention_mask"] = inputs["attention_mask"].cuda(device)
    max_length = max_to_generate + inputs["input_ids"].shape[1]
    if max_length > 2048:
        max_length = 2048
    return starcoder_family(inputs, max_length=max_length, top_p=top_p, temperature=temperature)

def codet5_generate(tokenizer, model, texts, max_to_generate, do_sample=None, top_p=None, temperature=None,
                    use_beam_search: bool = None, beam_size: int = None, device: str = None) -> List[str]:
    def codet5_family(encoding, top_p, temperature):
        with torch.no_grad():
            if use_beam_search:
                assert len(texts) == 1, "There is no need to use beam search with batch size > 1"
                outputs = model.generate(**encoding, max_length=max_to_generate,
                                         num_beams=beam_size, do_sample=False)
            else:
                if do_sample == False:  
                    outputs = model.generate(**encoding, max_length=max_to_generate, do_sample=False)
                else:
                    outputs = model.generate(**encoding, max_length=max_to_generate,
                                             top_p=top_p, temperature=temperature, do_sample=True)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return result

    encoding = tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048)
    encoding["input_ids"] = encoding["input_ids"].cuda(device)
    encoding["attention_mask"] = encoding["attention_mask"].cuda(device)
    # encoding['decoder_input_ids'] = encoding['input_ids'].clone()
    return codet5_family(encoding, top_p=top_p, temperature=temperature)


def opt_rerank(tokenizer, model, texts, device: str):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)

    inputs["input_ids"] = inputs["input_ids"].cuda(device)
    # print(inputs["input_ids"].shape)
    inputs["attention_mask"] = inputs["attention_mask"].cuda(device)
    with torch.no_grad():
        res = model(**inputs)
    # print(res[0])
    score_list = res[0].cpu().squeeze().tolist()
    if isinstance(score_list, float): 
        score_list = [score_list]
    return score_list


def codebert_rerank(tokenizer, model, texts, device: str):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    inputs["input_ids"] = inputs["input_ids"].cuda(device)
    # print(inputs["input_ids"].shape)
    # exit()
    inputs["attention_mask"] = inputs["attention_mask"].cuda(device)
    with torch.no_grad():
        res = model(**inputs)
    # print(res[0])
    score_list = res.logits.cpu().squeeze().tolist()
    if isinstance(score_list, float): 
        score_list = [score_list]
    return score_list


def starencoder_rerank(tokenizer, model, texts, device: str):
    """和codebert一样"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    inputs["input_ids"] = inputs["input_ids"].cuda(device)
    # print(inputs["input_ids"].shape)
    # exit()
    inputs["attention_mask"] = inputs["attention_mask"].cuda(device)
    with torch.no_grad():
        res = model(**inputs)
    # print(res[0])
    score_list = res.logits.cpu().squeeze().tolist()
    return score_list


def codet5_rerank(tokenizer, model, texts, device: str):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    inputs["input_ids"] = inputs["input_ids"].cuda(device)
    # print(inputs["input_ids"].shape)
    # exit()
    inputs["attention_mask"] = inputs["attention_mask"].cuda(device)
    with torch.no_grad():
        res = model(**inputs)
    # print(res[0])
    score_list = res.logits.cpu().squeeze().tolist()
    if isinstance(score_list, float): 
        score_list = [score_list]
    return score_list


def get_bleu(prediction: str, reference: str, max_order: int, smooth: bool, tokenizer=None, bleu=None, ):
    if len(prediction) == 0 or len(reference) == 0:
        return 0.0

    prediction = prediction.lower()
    reference = reference.lower()
    if tokenizer is None:
        tokenizer = Tokenizer13a()
    try:
        results = bleu.compute(predictions=[prediction], references=[[reference]],
                               tokenizer=tokenizer, max_order=max_order, smooth=smooth)
    except ZeroDivisionError as e:
        print(e)
        return 0.0
    else:
        # print(results)
        return results['bleu']


def get_rouge(prediction: str, reference: str, tokenizer=None, rouge=None):
    if len(prediction) == 0 or len(reference) == 0:
        return 0.0, 0.0, 0.0
    if tokenizer is None:
        results = rouge.compute(predictions=[prediction], references=[reference])
    else:
        results = rouge.compute(predictions=[prediction], references=[
            reference], tokenizer=tokenizer)
    # print(results)
    return results['rouge1'], results['rouge2'], results['rougeL']


def get_similarity(prediction: str, reference: str, tokenizer, model):
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Tokenize sentences
    sentences = [reference, prediction]
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')
    encoded_input["input_ids"] = encoded_input["input_ids"].cuda()
    encoded_input["attention_mask"] = encoded_input["attention_mask"].cuda()
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    cos_sim = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
    return cos_sim.item()


def get_meteor(prediction: str, reference: str, meteor):
    if len(prediction) == 0 or len(reference) == 0:
        return 0.0
    results = meteor.compute(predictions=[prediction], references=[reference])
    return results["meteor"]