"""Shared helpers for dialogue data construction."""

import time
from typing import Any, Dict, List, Optional, Sequence

import requests


SYSTEM_INSTRUCTION_EN = """Your task is to transform the given 'self-thinking process' and 'final answer',
together with the possible 'dialogue history', into a single round of interactive
Assistant-User dialogue, in order to enhance interactivity.

Rules:
1. Dialogue Generation
- Generate one round: Assistant asks, User answers.
- Must fully reflect the reasoning logic and conclusion.
- Do not add information absent from the reasoning/answer/history.

2. Questioning Rules
2.1 If reasoning shows ambiguity, multiple options, or missing info: ask the User.
2.2 If multiple options: let the User choose (must match final answer).
2.3 If missing info: ask the User to provide it.
2.4 No omissions: each gap = a question.

3. Expression Standards
3.1 Questions should be natural, human-like, and context-relevant.
3.2 After each question, provide a simulated User reply consistent with the final answer.
3.3 Dialogue must convey actual information, not empty agreement.
3.4 Format strictly:
  Assistant: ...
  User: ...
3.5 No fabrication beyond given reasoning/answer/history.
3.6 Dialogue must be concise, without redundancy or repetition, avoiding overlap with
    original reasoning, historical content, or self-content.

4. Dialogue History
4.1 Input may contain a dialogue history (history), or it may be empty.
4.2 If history exists, it must be referenced to ensure coherence with the new dialogue round.
4.3 If a question has been resolved in history, it should not be repeated; unresolved options
    may be followed up or prompted for user decision.
4.4 If history has covered necessary information, this round may omit related questions and
    focus on unresolved reasoning divergences or information gaps.
4.5 If no history exists, directly generate a new dialogue round based on the
    'self-thinking process' and 'final answer'.

5. Input & Output
5.1 Input includes: 'dialogue history' (history, may be empty or omitted),
    'self-thinking process' (question), and 'final answer' (answer).
5.2 Output is one round of Assistant-User dialogue that meets the above specifications.
"""

SYSTEM_INSTRUCTION_CN = """你的任务是将以下「自我思考过程」与「最终答案」，结合可能存在的「对话历史」，
改写为一轮主动的 Assistant-User 对话，以增强交互性。

请严格遵循以下规则：
1. 对话生成
1.1 你需要根据给定的思考过程及最终答案，参考对话历史（如有），生成一轮
    Assistant-User 对话（即 Assistant 提问，User 回答），以模拟模型与用户之间的互动推理过程。
1.2 对话内容必须完整体现原有的思考逻辑和结论，不得添加额外信息。

2. 提问规则
2.1 只要思考过程中出现推理分歧、多个选项或信息不足，Assistant 必须向用户发问。
2.2 如果思考过程遇到多个选项，应让用户选择一个，并确保用户的选择与最终答案一致。
2.3 若模型在思考中发现信息不足，也必须向用户提问，补全必要信息。
2.4 不允许省略提问的情况；每当推理涉及选择或信息补全，Assistant 均需通过提问引导用户作出决策或补充信息。

3. 表达规范
3.1 提问应自然、贴近人类表达习惯，并与上下文紧密衔接。
3.2 每次提问后，写出一条模拟用户回答。该回答只提供与既定结论一致的信息，不得影响后续推理，也不得改变最终答案。
3.3 对话应有实际信息增量，用户回答需体现判断或补充信息，避免无意义附和。
3.4 输出格式固定为一轮对话：Assistant: ...\\nUser: ...\\n。
3.5 不得编造原思考过程和最终答案中不存在的内容。
3.6 对话表达应精炼，无冗余、无重复，避免与原始推理内容、历史内容或自身内容重复。

4. 对话历史相关要求
4.1 输入可能包含一段对话历史（history），也可能没有历史，此时 history 为空或省略。
4.2 若存在历史，请充分参考历史内容，确保新一轮对话与历史上下文逻辑连贯、不重复。
4.3 如历史中已解决某一问题，本轮不再重复提问；如历史存在未决选项，可在本轮跟进追问或推动用户决策。
4.4 若历史已涵盖必要信息，本轮可省略相关提问，聚焦于尚未解决的推理分歧或信息缺口。
4.5 若无历史，则直接根据「自我思考过程」与「最终答案」生成新一轮对话。

5. 输入与输出
5.1 输入包括：「对话历史」（history，可为空或省略）、「自我思考过程」（question）和「最终答案」（answer）。
5.2 输出为一轮符合上述规范的 Assistant-User 对话。
"""

SENTENCE_END_TOKENS = {
    "\n\n",
    "。\n\n",
    "？\n\n",
    "！\n\n",
    ".\n\n",
    "?\n\n",
    "!\n\n",
    "<think>\n",
    " \n\n",
    "  \n\n",
    '."\n\n',
}

# Backward-compatible names used by older scripts.
system_instruction_en = SYSTEM_INSTRUCTION_EN
system_instruction_cn = SYSTEM_INSTRUCTION_CN


def format_history(history: Optional[Sequence[str]]) -> str:
    if not history:
        return "null"
    return "\n".join(history).strip()


def get_user_prompt(
    question: str,
    answer: str,
    history: Optional[Sequence[str]] = None,
    language: str = "en",
) -> str:
    history_str = format_history(history)

    if language == "cn":
        return (
            f"「自我思考过程」:\n{question.strip()}\n"
            f"「最终答案」:\n{answer.strip()}\n"
            f"「对话历史」:\n{history_str}"
        )

    return (
        f"'Self-thinking process':\n{question.strip()}\n"
        f"'Final answer':\n{answer.strip()}\n"
        f"'Dialogue history':\n{history_str}"
    )


def generate_question_think_template(
    tokenizer: Any,
    question: str,
    reasoning_content: str,
) -> List[int]:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"<think>\n{reasoning_content.strip()}\n\n"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        continue_final_message=True,
    )


def generate_question_template(tokenizer: Any, question: str) -> List[int]:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<think>\n"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        continue_final_message=True,
    )


def split_sentences_by_tokens(tokens: Sequence[str]) -> List[List[str]]:
    sentences: List[List[str]] = []
    current_sentence: List[str] = []

    for token in tokens:
        current_sentence.append(token)
        if token in SENTENCE_END_TOKENS:
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


class Assistant_Chat:
    """Small OpenAI-compatible chat client with dialogue history tracking."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        system_instruction: str = SYSTEM_INSTRUCTION_EN,
        language: str = "en",
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction
        self.language = language
        self.history: List[str] = []
        self.prompt_lengths: List[int] = []
        self.completion_lengths: List[int] = []

    def chat(
        self,
        question: str,
        answer: str,
        timeout: int = 360,
        retries: int = 3,
        retry_delay: int = 5,
    ) -> str:
        prompt = get_user_prompt(
            question=question,
            answer=answer,
            history=self.history,
            language=self.language,
        )
        self.prompt_lengths.append(len(prompt))

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                response_content = self._extract_content(response.json())
                self.history.append(f"{response_content}\n")
                self.completion_lengths.append(len(response_content))
                return response_content
            except requests.exceptions.Timeout:
                print(f"Attempt {attempt} of {retries}: Timeout.")
            except requests.exceptions.RequestException as exc:
                print(f"Attempt {attempt} of {retries}: {exc}")
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                print(f"Attempt {attempt} of {retries}: invalid response: {exc}")

            if attempt < retries:
                time.sleep(retry_delay)

        return f"Error: Failed after {retries} attempts."

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _extract_content(response_json: Dict[str, Any]) -> str:
        return response_json["choices"][0]["message"]["content"]
