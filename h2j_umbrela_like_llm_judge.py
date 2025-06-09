from umbrela.gpt_judge import GPTJudge
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrel', required=True, help='Path to the qrel file')
    parser.add_argument('--model_name', type=str, default='qwen3:8b', help='Model name to use for evaluation')
    parser.add_argument('--prompt_type', type=str, default='bing', help='Prompt type to use for evaluation')
    parser.add_argument('--base_url', type=str, default='http://localhost:11434/v1', help='Base URL for the LLM (Ollama)')
    args = parser.parse_args()
    
    
    print('Starting GPT judge')
    judge = GPTJudge(
    qrel=args.qrel, 
    model_name=args.model_name,
    prompt_type=args.prompt_type,
    base_url=args.base_url
    )


    print('Starting evaluation')
    judge.evalute_results_with_qrel(
            None,
            regenerate=True,
            num_samples=1,
            judge_cat=[0, 1, 2, 3],
    )