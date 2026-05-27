import json
from tqdm import tqdm
from .math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from transformers import AutoTokenizer
import os

class MathAnswerEvaluator:
    def __init__(self, dataset_path,tokenizer_path):
        """
        Initializes the MathAnswerEvaluator with the dataset file path.
        
        Args:
            dataset_path (str): Path to the dataset JSON file.
        """
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.rewards = []
        self.dataset_name = os.path.basename(self.dataset_path).replace(".json", "")

    def _load_dataset(self):
        """
        Loads the dataset from the given JSON file.

        Returns:
            list: A list of dataset entries.
        """
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)
        dataset = [item for item in dataset if 'output' in item]
        return dataset

    def _parse_solution(self, solution):
        """
        Parses the LaTeX solution string using the `parse` function.

        Args:
            solution (str): The LaTeX solution string to parse.
            extraction_mode (str): The mode to use for parsing.

        Returns:
            list: Parsed LaTeX solution as a list of expressions.
        """
        return extract_answer(solution)

    def _parse_answer(self, answer):
        """
        Parses the LaTeX answer string using a specific extraction configuration.

        Args:
            answer (str): The LaTeX answer string to parse.

        Returns:
            list: Parsed LaTeX answer as a list of expressions.
        """
        return extract_answer(answer)
    
    def _compute_token_length(self, response):
        """Computes token length using the tokenizer."""
        token_count = self.tokenizer.encode(response, return_tensors='pt')
        return token_count.shape[1]

    def _compute_reward(self, gold_parsed, answer_parsed):
        """
        Computes a binary reward by verifying the gold and answer expressions.

        Args:
            gold_parsed (list): Parsed gold solution as a list of expressions.
            answer_parsed (list): Parsed answer as a list of expressions.

        Returns:
            float or None: Binary reward (1.0 or 0.0) if verifiable, or None if verification fails.
        """

        try:
            return grade_answer_mathd(answer_parsed, gold_parsed) or grade_answer_sympy(answer_parsed, gold_parsed)
        except Exception as e:
            print(f"Verification failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        return False

    def evaluate(self):
        """
        Evaluates the dataset, computes rewards for each entry, and calculates accuracy.

        Returns:
            dict: A dictionary containing the rewards and the overall accuracy.
        """
        rewards = []
        token_lengths = []
        for item in tqdm(self.dataset):
            con = item['output']
            sol = item['solution']
            token_length = self._compute_token_length(con)
            token_lengths.append(token_length)

            # Parse the gold solution
            gold_parsed = self._parse_solution(sol)

            # Parse the answer and compute reward if valid
            if "</think>" in con:
                con = con.split("</think>")[-1]
                answer_parsed = self._parse_answer(con)
                reward = self._compute_reward(gold_parsed, answer_parsed)
                rewards.append(reward)
            else:
                # rewards.append(False)
                con = con
                answer_parsed = self._parse_answer(con)
                reward = self._compute_reward(gold_parsed, answer_parsed)
                rewards.append(reward)

        # Calculate accuracy as the mean of valid rewards
        accuracy = sum(rewards) / len(rewards) if rewards else 0.0
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
        print("===== Math Answer Evaluation Results =====")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Average Token Length: {avg_token_length:.2f}")

        # Prepare final result
        return {
            "accuracy": accuracy,
            "avg_token_length": avg_token_length,
            "rewards": rewards,
            "token_lengths": token_lengths
        }

    def save_results(self, output_path):
        """
        Saves the evaluation results (including accuracy) to a JSON file.

        Args:
            output_path (str): Path to save the results JSON file.
        """
        results = self.evaluate()
        output_path = os.path.join(os.path.dirname(output_path), f"{self.dataset_name}_eval_result.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    evaluator = MathAnswerEvaluator(dataset_path="/home/chenxin/verl-interactive/real_interaction_app/results/Proactive-Interactive-R1-Math-7B_human_interactive_test.parquet_question_generation_result.json", tokenizer_path="/home/chenxin/shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    evaluator.evaluate()
    evaluator.save_results("results/")