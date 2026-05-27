import string
import re
import collections
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
from ..utils.model_client import ModelClient
from ..utils.simulator_client import UserSimulatorClient
from ..utils.extract_json_reliable import extract_json
import litellm
from openai import OpenAI

system_prompt = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations."
    )

def load_coqa_dataset(args):
    """
    加载 squad rc.wikipedia 的 validation 集。
    """
    print("正在加载数据集...")

    # dataset = load_dataset(args.data_dir, split="validation")
    # return dataset
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    # print(dataset['data'][0]['target_turn']['question'])
    return dataset['data']

def query_llm(prompt, args):
    """
    调用 LLM 接口，使用 argparse 传入的参数。
    """
    try:
        # 使用 args 中的参数初始化 ModelClient
        model_client = ModelClient(
            model_path=args.model_name,
            base_url=args.model_url,
            stop_tokens=["<｜end▁of▁sentence｜>"],
            reasoning_model=args.reasoning_model
        )
        
        # 根据参数决定是否添加 system prompt (虽然你的 args 里有 flag 但没给具体 prompt 内容，这里留个位置)
        messages = []
        if args.reasoning_while_asking_sys_prompt:
             messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # 注意：这里假设 model_client.chat 返回响应内容，或者更新内部状态
        # 根据你原本的代码逻辑，似乎是调用 chat 后访问 .completion
        model_response = model_client.chat(messages=messages)
        
        final_response = model_client.completion
        
        if args.reasoning_model:
             output = final_response.split("</think>")[-1].strip()
        else:
             output = final_response
        return output,final_response
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error {e}", f"Error {e}"

def process_single_item(item_data):
    """
    处理单条数据。
    item_data 是一个元组: (index, item, args)
    """
    i, item, args = item_data
    question = item['target_turn']['question']
    
    # 获取所有可能的标准答案列表
    # normalized_ground_truths = item['answer']['normalized_aliases'] 
    ground_truths = item['target_turn']['answer']
    ambiguity = item['ambiguity']
    story = item['story']
    
    # --- 构建 Zero-shot Prompt ---
    prompt = f"Can you help me answer a question about the following story?\n\n{story}\n\nMy question is: {question}"
    
    # --- 调用模型 ---
    output, final_response = query_llm(prompt, args)

    # 构建结果字典
    result = {
        "index": i,
        "question": question,
        "prompt": prompt,
        "prediction": output,
        "final_response": final_response,
        "ground_truths": ground_truths,
        "ambiguity": ambiguity
    }

    return result

def evaluate_coqa(args):
    print(args)
    # 1. 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_file = os.path.join(args.output_dir, f"coqa_results_{args.model_name}.jsonl")
    print(f"结果将保存至: {output_file}")

    # 2. 加载数据
    dataset = load_coqa_dataset(args)

    print(f"开始评测，共 {len(dataset)} 个样本，并发数: {args.num_workers}...")

    results = []
    
    # 3. 准备任务列表
    tasks = [(i, item, args) for i, item in enumerate(dataset)]

    # 4. 并发执行
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 使用 tqdm 显示进度
        futures = {executor.submit(process_single_item, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task failed: {e}")


    print("\n" + "="*30)
    print(f"Final Results ({len(results)} samples processed):")
    print("="*30)

    # 5. 保存详细结果
    results.sort(key=lambda x: x['index']) # 按索引排序
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="/home/chenxin/verl-interactive/generalization_eval/coqa")
    parser.add_argument("--model_name",type=str,default="Proactive-Interactive-R1-Math-7B-new")
    parser.add_argument("--model_url",type=str,default="http://localhost:1137")
    parser.add_argument("--reasoning_model", action="store_true",
                        help="Whether to use the reasoning model setup")
    parser.add_argument("--num_workers", "-n", type=int, default=64,
                       help="Number of concurrent queries")
    parser.add_argument("--reasoning_while_asking_sys_prompt", action = "store_true",
                        help="Add system prompt for reasoning")
    parser.add_argument("--data_dir", type=str, default="/home/chenxin/verl-interactive/datasets/Abg-CoQA/coqa_abg_test.json")
    args = parser.parse_args()
    args.reasoning_model = True # 这里暂时设置为 True，实际使用时根据需要调整
    args.reasoning_while_asking_sys_prompt = True # 这里暂时设置为 True，实际使用时根据需要调整
    evaluate_coqa(args)