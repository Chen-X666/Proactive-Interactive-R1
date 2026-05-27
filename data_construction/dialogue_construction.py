"""
Build dialogue-augmented reasoning data.

Pipeline:
1. Load a JSON/JSONL dataset and slice it by ``idx_from``/``idx_to``.
2. Render the question and reasoning into the model chat template.
3. Split the reasoning into token-level sentence chunks.
4. Score each sentence with average negative log probability.
5. For high-uncertainty sentences, call the dialogue API and insert one
   Assistant-User round after that sentence.
6. Save the accumulated results after each successfully processed item.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    Assistant_Chat,
    generate_question_template,
    generate_question_think_template,
    split_sentences_by_tokens,
)


DEFAULT_API_URL = "https://api.ai-gaochao.cn/v1/"
DEFAULT_API_MODEL = "gpt-4o-2024-11-20"
SPECIAL_TOKENS = ["<asking>", "</asking>", "<response>", "</response>"]
SKIPPED_RESPONSE = "Skipped due to low uncertainty"
CHAT_COMPLETIONS_PATH = "chat/completions"
ASSISTANT_RE = re.compile(r"Assistant:\s*(.*?)(?:\n\n|$)", re.DOTALL)
USER_RE = re.compile(r"User:\s*(.*?)(?:\n\n|$)", re.DOTALL)


def read_dataset(input_file: str) -> List[Dict[str, Any]]:
    path = Path(input_file)
    with path.open("r", encoding="utf-8") as file:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in file if line.strip()]

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset in {input_file}, got {type(data).__name__}.")

    return data


def output_file_path(input_file: str, output_path: str, uncertainty_threshold: float) -> Path:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(input_file).stem
    return output_dir / f"{file_name}_uncertainty_threshold_{uncertainty_threshold}_full_dataset.json"


def normalize_api_url(api_url: str) -> str:
    return api_url.rstrip("/") + "/" + CHAT_COMPLETIONS_PATH


def get_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or os.getenv("DIALOGUE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is required. Pass --api_key or set DIALOGUE_API_KEY/OPENAI_API_KEY."
        )
    return api_key


def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    original_vocab_size = len(tokenizer)
    added_token_count = tokenizer.add_tokens(SPECIAL_TOKENS)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if added_token_count:
        model.resize_token_embeddings(len(tokenizer))

    print("原来的 vocab 大小:", original_vocab_size)
    print("新增加的 token 数量:", added_token_count)
    print("现在的 vocab 大小:", len(tokenizer))

    return tokenizer, model


def build_initial_inputs(
    tokenizer: Any,
    model: Any,
    question: str,
    reasoning_content: str,
) -> Tuple[List[int], Any, str, int, List[List[str]]]:
    input_ids = generate_question_think_template(tokenizer, question, reasoning_content)
    inputs = tokenizer.pad({"input_ids": [input_ids]}, return_tensors="pt").to(model.device)
    input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    prompt_length = len(generate_question_template(tokenizer=tokenizer, question=question))
    reasoning_token_ids = input_ids[prompt_length:]
    sentence_tokens = split_sentences_by_tokens(
        [tokenizer.decode(tid, skip_special_tokens=True) for tid in reasoning_token_ids]
    )

    return input_ids, inputs, input_text, prompt_length, sentence_tokens


def sentence_uncertainty(
    model: Any,
    inputs: Any,
    sentence_token_ids: Sequence[int],
    start_token_idx: int,
) -> float:
    with torch.no_grad():
        outputs = model(**inputs)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

    token_log_probs = [
        -log_probs[0, start_token_idx + idx, int(token_id)].item()
        for idx, token_id in enumerate(sentence_token_ids)
    ]
    return sum(token_log_probs) / len(token_log_probs)


def parse_dialogue_response(response: str) -> Tuple[str, str]:
    assistant_match = ASSISTANT_RE.search(response)
    user_match = USER_RE.search(response)

    if not assistant_match or not user_match:
        raise ValueError(f"无法提取完整对话: {response}")

    assistant_text = assistant_match.group(1).strip()
    user_text = user_match.group(1).strip()
    return assistant_text, user_text


def replacement_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def extract_reasoning_after_think(input_text: str) -> str:
    if "<think>" not in input_text:
        return input_text.strip()
    return input_text.split("<think>", 1)[1].strip()


def save_json(data: List[Dict[str, Any]], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def process_item(
    item: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    assistant_chat: Assistant_Chat,
    uncertainty_threshold: float,
    max_replacements_per_item: int,
) -> Dict[str, Any]:
    question = item["input"]
    reasoning_content = item["reasoning_content"]
    input_ids, inputs, input_text, start_token_idx, sentences = build_initial_inputs(
        tokenizer,
        model,
        question,
        reasoning_content,
    )

    sentence_list: List[str] = []
    sentence_pe: List[float] = []
    sentence_responses: List[str] = []
    replaced_count = 0

    for sent_tokens in sentences:
        end_token_idx = start_token_idx + len(sent_tokens)
        sent_text = tokenizer.decode(
            input_ids[start_token_idx:end_token_idx],
            skip_special_tokens=False,
        )

        if not sent_text:
            start_token_idx += len(sent_tokens)
            continue

        sent_token_ids = input_ids[start_token_idx:end_token_idx]
        pe = sentence_uncertainty(model, inputs, sent_token_ids, start_token_idx)
        sentence_pe.append(pe)
        sentence_list.append(sent_text)

        if pe <= uncertainty_threshold:
            sentence_responses.append(SKIPPED_RESPONSE)
            start_token_idx += len(sent_tokens)
            continue

        if replaced_count >= max_replacements_per_item:
            raise ValueError(
                f"Reached maximum replacements of {max_replacements_per_item}, skipping this data."
            )

        response = assistant_chat.chat(question=sent_text, answer=item["content"])
        sentence_responses.append(response)

        assistant_text, user_text = parse_dialogue_response(response)
        asking_response_sentence = (
            f"<asking>{assistant_text}</asking>\n"
            f"<response>{user_text}</response>\n\n"
        )
        replaced_sentence = sent_text + asking_response_sentence

        input_text = input_text.replace(sent_text, replaced_sentence, 1)
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        input_ids = inputs["input_ids"][0]

        start_token_idx += replacement_token_count(tokenizer, replaced_sentence)
        replaced_count += 1

    processed_item = dict(item)
    processed_item["sentences_info"] = {
        "sentences": sentence_list,
        "sentence_pe": sentence_pe,
        "sentence_response": sentence_responses,
    }
    processed_item["reasoning_content_ask_rep"] = extract_reasoning_after_think(input_text)
    processed_item["prompt_lengths"] = assistant_chat.prompt_lengths
    processed_item["completion_lengths"] = assistant_chat.completion_lengths

    torch.cuda.empty_cache()
    return processed_item


def main(args: argparse.Namespace) -> None:
    api_key = get_api_key(args)
    api_url = normalize_api_url(args.api_url)
    output_file = output_file_path(
        args.input_file,
        args.output_path,
        args.uncertainty_threshold,
    )

    tokenizer, model = load_model_and_tokenizer(args.model_path)
    dataset = read_dataset(args.input_file)[args.idx_from : args.idx_to]

    full_datasets: List[Dict[str, Any]] = []
    error_count = 0

    with tqdm(total=len(dataset), desc="Processing") as pbar:
        for item in dataset:
            try:
                assistant_chat = Assistant_Chat(
                    api_url=api_url,
                    api_key=api_key,
                    model=args.api_model,
                )
                processed_item = process_item(
                    item=item,
                    tokenizer=tokenizer,
                    model=model,
                    assistant_chat=assistant_chat,
                    uncertainty_threshold=args.uncertainty_threshold,
                    max_replacements_per_item=args.max_replacements_per_item,
                )
                full_datasets.append(processed_item)
                save_json(full_datasets, output_file)
            except Exception as exc:
                error_count += 1
                item_input = item.get("input", "<missing input>")
                print(f"Error processing item with input: {item_input}. Error: {exc}")
            finally:
                pbar.update(1)

    print(f"Total errors encountered: {error_count}")
    print(f"Total full datasets: {len(full_datasets)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentence Uncertainty Dialogue Pipeline")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data1/HF-Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Path to Hugging Face model",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/chenxin/proactive_interactive_r1/dataset_construction/distill_r1_coig_neo_en_cleaned.json",
        help="Input JSON/JSONL file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=18.73,
        help="Threshold for sentence uncertainty to trigger dialogue generation",
    )
    parser.add_argument(
        "--idx_from",
        type=int,
        default=0,
        help="Start index for processing the dataset",
    )
    parser.add_argument(
        "--idx_to",
        type=int,
        default=10000,
        help="End index for processing the dataset",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=DEFAULT_API_URL,
        help="Base URL for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key. If omitted, DIALOGUE_API_KEY or OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default=DEFAULT_API_MODEL,
        help="API model used for dialogue generation",
    )
    parser.add_argument(
        "--max_replacements_per_item",
        type=int,
        default=10,
        help="Maximum high-uncertainty sentence replacements per item",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
