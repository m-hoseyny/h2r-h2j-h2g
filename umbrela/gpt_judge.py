import argparse
import os
from typing_extensions import Optional

from dotenv import load_dotenv
import openai
from openai import AzureOpenAI, OpenAI
from retry import retry
from tqdm import tqdm

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class GPTJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 0,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(qrel, model_name, prompt_file, prompt_type, few_shot_count)
        self.create_openai_client(base_url)

    def create_openai_client(self, base_url: Optional[str] = None):
        api_key = os.environ.get("OPEN_AI_API_KEY", 'ollama')
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        azure_endpoint = os.environ.get("AZURE_OPENAI_API_BASE")

        if all([api_key, azure_endpoint, api_version]):
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
            self.use_azure_ai = True
            self.engine = os.environ["DEPLOYMENT_NAME"]
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.engine = self.model_name
            self.use_azure_ai = False

    @retry(tries=3, delay=0.1)
    def run_gpt(self, prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0,
            )
            output = (
                response.choices[0].message.content.lower()
                if response.choices[0].message.content
                else ""
            )
        except openai.BadRequestError as e:
            print(f"Encountered {e} for {prompt}")
            output = ""
        return output

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
        prepocess: bool,
    ):
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )

        outputs = [
            self.run_gpt(prompt, max_new_tokens) for prompt in tqdm(self.prompts)
        ]
        return outputs

    def judge(self, request_dict, max_new_tokens=100, prepocess: bool = True):
        outputs = self.predict_with_llm(request_dict, max_new_tokens, prepocess)
        return common_utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel", type=str, help="qrels file", required=True)
    parser.add_argument("--result_file", type=str, help="retriever result file")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument(
        "--prompt_type", type=str, help="Prompt type. Supported types: [bing, basic]."
    )
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument(
        "--few_shot_count", type=int, help="Few shot count for each category."
    )
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--regenerate", action="store_true")

    args = parser.parse_args()
    load_dotenv()

    judge = GPTJudge(
        args.qrel, args.model, args.prompt_file, args.prompt_type, args.few_shot_count
    )
    judge.evalute_results_with_qrel(
        args.result_file,
        regenerate=args.regenerate,
        num_samples=args.num_sample,
        judge_cat=JUDGE_CAT,
    )


if __name__ == "__main__":
    main()
