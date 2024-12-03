from typing import *


def replace_subsum_to_modelgen_infer_res_format(original_samples: List[Dict], data_include_new_fields: Dict):
    # print(len(original_samples), len(infer_samples))
    assert "allResult" in data_include_new_fields
    all_result = data_include_new_fields["allResult"]
    assert len(original_samples) == len(all_result)
    for i in range(len(original_samples)):
        original_sample = original_samples[i]
        infer_sample = all_result[i]
        assert original_sample["only_code"] == infer_sample["code"]
        assert "small_sum" in original_sample and "all_gen_res" in infer_sample \
            and len(infer_sample["all_gen_res"]) > 0
        original_sample["small_sum"] = infer_sample["all_gen_res"][0]


def replace_subsum_to_modelgen_origin_file_format(original_samples: List[Dict], data_include_new_fields: List[Dict]):
    # print(len(original_samples), len(infer_samples))
    assert len(original_samples) == len(data_include_new_fields)
    for i in range(len(original_samples)):
        original_sample = original_samples[i]
        infer_sample = data_include_new_fields[i]
        assert original_sample["only_code"] == infer_sample["code_wo_docstring"]
        assert "small_sum" in original_sample and "model-sub-summary" in infer_sample
        original_sample["small_sum"] = infer_sample["model-sub-summary"]


replace_rules = {
    "replace_subsum_infer_res_format": replace_subsum_to_modelgen_infer_res_format,
    "replace_subsum_origin_file_format": replace_subsum_to_modelgen_origin_file_format
}
