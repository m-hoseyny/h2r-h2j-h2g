import pandas as pd
import random
import os
import re
from datetime import datetime
import argparse
import openai


client = openai.OpenAI(api_key='ollama', 
                           base_url='http://localhost:11434/v1')


prompts = """Write a passage that answers the following query: {}"""

# Read the queries file

def read_queries(queries_path):
    queries_df = pd.read_csv(queries_path, sep='\t', header=None, names=['id', 'query'])
    return queries_df

def make_prompt(row):
    return prompts.format(row['query'])


def generate_passages(queries, model):
    results = []
    for _, row in tqdm.tqdm(queries.iterrows(), total=len(queries)):
        messages = [
            {
                'role': 'system',
                'content': 'You are a passage generator.'
            },
            {
            'role': 'user',
            'content': make_prompt(row)
        }
        ]
        
        response = client.chat.completions.create(
            model=model, 
            messages=messages, 
            temperature=0.7,
            top_p=1
        )
        
        full_response = response.choices[0].message.content
        # Extract answer using regex
        answer = None
        # Pattern matches 'Answer:', 'Answer :', or variations with different cases
        answer_pattern = re.compile(r'(?i)answer\s*:\s*(.*?)(?=\n|$)')
        match = answer_pattern.search(full_response)
        if match:
            answer = match.group(1).strip()
        
        results.append({
            'query_id': row['id'],
            'query': row['query'],
            'response': full_response,
            'extracted_answer': answer  # Will be None if no answer found
        })
        
    return results


def write_passages(passages, output):
    print(f"Writing passages to {output}")
    pd.DataFrame(passages).to_json(output, orient='records', lines=True)

def main():
    parser = argparse.ArgumentParser(description='Generate passages for the given queries')
    parser.add_argument('--queries', required=True, help='Path to the queries file (TSV format with id and query columns)')
    parser.add_argument('--output', required=True, help='Path to the output passages file')
    parser.add_argument('--model', type=str, default='qwen3:8b', help='Model to use for generation (default: qwen3:8b)')
    
    args = parser.parse_args()

    
    # Read queries
    queries = read_queries(args.queries)
    
    print(f"Read {len(queries)} queries")
    
    # Generate passages
    passages = generate_passages(queries, args.model)
    
    print(f"Generated {len(passages)} passages")
    
    # Write passages to output file
    write_passages(passages, args.output)
    
    print(f"Wrote passages to {args.output}")


if __name__ == '__main__':
    main()


