"""
Interactive-R1 style Code Exact Match Reward Manager
"""
import torch
import random
import regex as re
import json
import time
import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from verl import DataProto
from verl.workers.reward_manager.registry import register
from .reward_score import _default_compute_score
from collections import defaultdict
from pathlib import Path
from .metrics.helpfulness import HelpfullnessMetric
from bigcodebench.eval import untrusted_check

# Global lock to keep concurrent debug printing readable
PRINT_LOCK = threading.Lock()

GLOBAL_IO_EXECUTOR = ThreadPoolExecutor(max_workers=48)


def _extract_code_blocks(text: str) -> str:
    """
    Extract code blocks from the given text.
    Code blocks are defined as text enclosed within triple backticks (```).
    """
    # 模式1: 匹配 ```python ... ``` (最优先)
    pattern_python = r"```python\n(.*?)```"
    match = re.search(pattern_python, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # 模式2: 匹配 ``` ... ``` (如果没写 python 标签)
    pattern_generic = r"```\n(.*?)```"
    match = re.search(pattern_generic, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # 如果没有 markdown 标记，通常直接返回原始内容，或者做进一步清洗
    return text

def _pass_rate_check(completion, metadata) -> bool:
    """
    Check if the answer passes untrusted check.
    """
    res = untrusted_check(
            completion,
            metadata["test"],
            metadata["entry_point"],
            max_as_limit=300 * 1024,
            max_data_limit=300 * 1024,
            max_stack_limit=300 * 1024,
            min_time_limit=60,
            gt_time_limit=60
        )
    passed, info = res[0] == "pass", res[1]
    return float(passed), info


def _count_asking_tags(text):
    opening_tags = text.count("<asking>")
    closing_tags = text.count("</asking>")
    return opening_tags, closing_tags

def _extract_qa_pairs(think_inner):

    if "<asking>" not in think_inner or "<response>" not in think_inner or "</asking>" not in think_inner or "</response>" not in think_inner:
        return False, ""

    pattern_full = r"^(?:\s*\S.*?<asking>\s*\S.*?</asking>\s*<response>\s*\S.*?</response>\s*.*?)*$"

    if re.match(pattern_full, think_inner, re.DOTALL):
        pairs = re.findall(
            r"<asking>\s*(\S.*?)\s*</asking>\s*<response>\s*(\S.*?)\s*</response>",
            think_inner,
            re.DOTALL,
        )

        return True, pairs

    return False, ""


def _compute_helpfullness_reward(question, response, qa_pairs):
    helpfullness_metric = HelpfullnessMetric()
    helpfulness_reward = helpfullness_metric.score(
        question=question,
        response=response,
        qa_pairs=qa_pairs,

    )
    helpfulness_goldens = float(helpfulness_reward)
    return helpfulness_goldens


def compute_score(solution_str, ground_truth):
    """
    The scoring function for Interactive-R1 style exact match (EM).
    """
    if "</think>" in solution_str.split("<｜Assistant｜>")[-1]:
        model_think = solution_str.split("<｜Assistant｜>")[-1].split("</think>")[0]
        model_output = solution_str.split("<｜Assistant｜>")[-1].split("</think>")[1]

    else:
        model_output = ""
        model_think = ""

    extract_code_result = _extract_code_blocks(model_output)
    open_count, close_count = _count_asking_tags(model_think)

    do_print = random.randint(1, 64) == 1

    if do_print:
        with PRINT_LOCK:
            print("--------------------------------")
            # ground truth
            print(f"Golden answers: {ground_truth.get('target', ground_truth) if isinstance(ground_truth, dict) else ground_truth}")

            # extracted answer from model
            if extract_code_result is not None:
                print(f"Extracted answer is not None: {extract_code_result}")
            else:
                print("Extracted answer: None!")

            # raw output of the model
            print(f"Solution string: {solution_str}")

    if extract_code_result is None:
        return 0
    else:
        single_turn_prompt_raw = ground_truth.get('single_turn_prompt', "") if isinstance(ground_truth, dict) else ""

        metadata = ground_truth.get('single_turn_metadata', {}) if isinstance(ground_truth, dict) else {}
        pass_rate_check, _ = _pass_rate_check(extract_code_result.strip(), metadata)

        judge, qa_pairs = _extract_qa_pairs(model_think)

        if pass_rate_check:
            if open_count > 0 or close_count > 0:
                if judge and open_count == close_count:
                    eff_reward = (5 - open_count) / (5 - 1)
                    
                    helpful_reward = []

                    futures_map = {}
                    for i, (current_asking, current_response) in enumerate(qa_pairs):
                        prev_history = qa_pairs[:i]
                        future = GLOBAL_IO_EXECUTOR.submit(
                            _compute_helpfullness_reward,
                            question=single_turn_prompt_raw,
                            response=current_asking,
                            qa_pairs=prev_history
                        )
                        futures_map[future] = i

                    for f in as_completed(futures_map):
                        helpful_reward.append(f.result())

                    return 0.5 + 0.5 * np.mean(helpful_reward) * eff_reward
                else:
                    return 0.5
            else:
                return 0.5
        else:
            return 0.0


@register("interactive_r1_code_collab")
class InteractiveR1CodeRewardManager:
    name = "interactive_r1_code_collab"
    
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, format_score=0.0, score=1.0, run_id=None, **kwargs) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer
            model_path = "/data1/HF-Models/Qwen/Qwen2.5-7B-Instruct" 
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                tokenizer = None
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.format_score = format_score
        self.score = score
        self.step = None
        self.run_id = run_id

    def _setup_record_dir(self, data):
        if not hasattr(self, 'record_dir'):
            base_parent = Path(__file__).parent.parent.parent.parent / "verl_step_records"
            if hasattr(self, 'run_id') and self.run_id:
                self.record_dir = base_parent / self.run_id
            else:
                self.record_dir = base_parent / f"torl-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
            self.record_dir.mkdir(parents=True, exist_ok=True)

        if self.step is None:
            max_step = -1
            if self.record_dir.exists():
                for file in os.listdir(self.record_dir):
                    match = re.search(r"step(?:-val)?-(\d+)\.json", file)
                    if match:
                        step_idx = int(match.group(1))
                        if step_idx > max_step:
                            max_step = step_idx
            self.step = max_step + 1
        
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

    def __call__(self, data: DataProto, return_dict=False):
        save_record = data.meta_info.get('save_record', True)
        self._setup_record_dir(data)

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # 用于临时存储结果的列表，确保索引对应
        batch_results = [None] * batch_size
        already_print_cnt = 0

        # --- Worker Function ---
        def process_item(i, data_item):
            try:
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                if 'reward_model' in data_item.non_tensor_batch:
                    ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                else:
                    ground_truth = data_item.non_tensor_batch.get('ground_truth', 
                                data_item.non_tensor_batch.get('golden_answers', []))
                
                data_source = data_item.non_tensor_batch.get('data_source', 'unknown')

                # 调用计算函数
                calculated_score = compute_score(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                )

                record = {
                    'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                    'data_source': data_source,
                    "prompt": self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False),
                    "response": self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False),
                    'ground_truth': ground_truth,
                    'score': calculated_score,
                    'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
                    'extra_info': data_item.non_tensor_batch.get('extra_info', None),
                }
                
                if "turns_stats" in data_item.non_tensor_batch:
                    record['num_turn'] = data_item.non_tensor_batch["turns_stats"]
                    record['num_valid_action'] = data_item.non_tensor_batch["valid_action_stats"]
                    record['is_done'] = not data_item.non_tensor_batch["active_mask"]

                return i, calculated_score, valid_response_length, record

            except Exception as e:
                print(f"[Batch Error] Item {i}: {e}")
                return i, 0.0, 1, None

        # --- 并行执行 ---
        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count() + 4)) as executor:
            # 提交任务
            futures = [executor.submit(process_item, i, data[i]) for i in range(batch_size)]
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    i, score, valid_len, record = future.result()
                    batch_results[i] = (score, valid_len, record)
                except Exception as e:
                    print(f"Future Error: {e}")

        # --- 关键：按原始顺序组装结果 (Strict Order Assembly) ---
        to_save_records = []
        for i in range(batch_size):
            res = batch_results[i]
            
            # 如果某个线程彻底失败，填充默认值
            if res is None:
                score, valid_len, record = 0.0, 1, None
            else:
                score, valid_len, record = res
            
            # 1. 填充 Tensor
            idx_pos = max(0, int(valid_len) - 1)
            idx_pos = min(idx_pos, reward_tensor.shape[1] - 1)
            reward_tensor[i, idx_pos] = score
            
            # 2. 填充 extra_info (必须有序！)
            reward_extra_info['score'].append(score)
            
            if score > 0:
                reward_extra_info['correct_response_length'].append(valid_len)
            else:
                reward_extra_info['wrong_response_length'].append(valid_len)
            
            # 3. 收集记录
            to_save_records.append(record)
            
            # 调试打印
            if already_print_cnt < self.num_examine:
                already_print_cnt += 1
                print(f"=== Debug Item {i} ===")
                print(f"Score: {score}")
                print("=" * 30)

        # --- 保存记录 ---
        if save_record:
            filename = f"{self.name}-step-val-{self.step}.json" if self.num_examine == 1 else f"{self.name}-step-{self.step}.json"
            save_path = self.record_dir / filename
            valid_records = [r for r in to_save_records if r is not None]
            try:
                if save_path.exists():
                    with open(save_path, "r") as f:
                        existing = json.load(f)
                    existing.extend(valid_records)
                    with open(save_path, "w") as f:
                        json.dump(existing, f, indent=4)
                else:
                    with open(save_path, "w") as f:
                        json.dump(valid_records, f, indent=4)
            except Exception as e:
                print(f"Save Error: {e}")

        self.step += 1

        # 计算平均长度统计
        c_len = reward_extra_info.get('correct_response_length', [])
        w_len = reward_extra_info.get('wrong_response_length', [])
        mean_c = float(np.mean(c_len)) if c_len else 0.0
        mean_w = float(np.mean(w_len)) if w_len else 0.0
        
        # 广播回列表长度
        reward_extra_info['correct_response_length'] = [mean_c] * batch_size
        reward_extra_info['wrong_response_length'] = [mean_w] * batch_size

        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor