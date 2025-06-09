import pandas as pd
import os, sys
import argparse
import copy
import ast
import openai
import time
import tqdm

client = openai.OpenAI(api_key='ollama', 
                           base_url='http://localhost:11434/v1')


Hard2RerieveMessage = [
                {
                    "role": "system",
                    "content": (
                        "You are AssessmentLLM, an intelligent assistant that can update a list of assessment criteria \
to best provide all the atomic reasons that why this query is hard to retrieve"
                    )
                },
                {
                    "role": "user",
                    "content": ( 'Update the list of atomic reasons (1-12 words), if needed, '
                    'so they best provide the atomic reasons that why the query is hard to retrieve.'
                    'Leverage only the initial list of atomic reasons (if exists)'
                    'Return only the final list of all atomic reasons in a Pythonic list format (even if no updates). '
                    'Make sure there is no redundant atomic reasons. '
                    'Ensure the updated atomic reasons list has at most 30 atomic reasons (can be less), keeping only the most vital ones. '
                    'Order them in decreasing order of importance. '
                    'Prefer atomic reasons that provide more information about why the query is hard to retrieve. \n '
                    'Search Query: {} '
                    # 'Context: ' + ' '.join(context) +
                    # f'Search Query: {query} \n '
                    'Initial Assessment Criteria List: {} \n '
                    'Initial Assessment Criteria List Length: {} \n '
                    'Only update the list of atomic reasons (if needed, else return as is). Do not explain. '
                    'Always answer in short atomic reasons (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ". '
                    'Updated Assessment Criteria List:')
                }
                ]

Hard2GenerateMessage = [
                {
                    "role": "system",
                    "content": (
                        "You are AssessmentLLM, an intelligent assistant that provides a list of assessment criteria \
to best provide all the atomic reasons that why this query would be hard to generate a passage to answer the query"
                    )
                },
                {
                    "role": "user",
                    "content": ( 'Provide a list of atomic reasons (1-12 words) '
                    'so they best provide the atomic reasons that why the query is hard to generate a passage to answer the query.'
                    # 'Leverage only the initial list of atomic reasons (if exists)'
                    'Return only the final list of all atomic reasons in a Pythonic list format (even if no updates). '
                    'Make sure there is no redundant atomic reasons. '
                    'Ensure the atomic reasons list has between 10 and 30 reasons, keeping only the most vital ones. '
                    'Order them in decreasing order of importance. '
                    'Prefer atomic reasons that provide more information about why the query would be hard to generate a passage to answer the query. \n '
                    'Search Query: {} '
                    # 'Context: ' + ' '.join(context) +
                    # f'Search Query: {query} \n '
                    # 'Initial Assessment Criteria List: {} \n '
                    # 'Initial Assessment Criteria List Length: {} \n '
                    # 'Only update the list of atomic reasons (if needed, else return as is). Do not explain. '
                    'Always answer in short atomic reasons (not questions). List in the form ["reason 1", "reason 2", ...] and reason 1 and reason 2 are strings with no mention of ". '
                    'Assessment Criteria List:')
                }
                ]


Hard2Judge = [
                {
                    "role": "system",
                    "content": (
                        "You are AssessmentLLM, an intelligent assistant that can update a list of assessment criteria \
to best provide all the atomic reasons that why this query is hard to judge by an LLM based on the query and its relevant passages"
                    )
                },
                {
                    "role": "user",
                    "content": ( 'Update the list of atomic reasons (1-12 words), if needed, '
                    'so they best provide the atomic reasons that why the query is hard to judge by an LLM based on the query and its relevant passages.'
                    'Leverage only the initial list of atomic reasons (if exists)'
                    'Return only the final list of all atomic reasons in a Pythonic list format (even if no updates). '
                    'Make sure there is no redundant atomic reasons. '
                    'Ensure the updated atomic reasons list has at most 30 atomic reasons (can be less), keeping only the most vital ones. '
                    'Order them in decreasing order of importance. '
                    'Prefer atomic reasons that provide more information about why the query is hard to judge by an LLM based on the query and its relevant passages. \n '
                    'Search Query: {} '
                    # 'Context: ' + ' '.join(context) +
                    # f'Search Query: {query} \n '
                    'Initial Assessment Criteria List: {} \n '
                    'Initial Assessment Criteria List Length: {} \n '
                    'Only update the list of atomic reasons (if needed, else return as is). Do not explain. '
                    'Always answer in short atomic reasons (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ". '
                    'Updated Assessment Criteria List:')
                }
                ]



def recursive_reasoning(df, task):
    

    if task == 'h2r':
        messages = Hard2RerieveMessage
    elif task == 'h2g':
        messages = Hard2GenerateMessage
    elif task == 'h2j':
        messages = Hard2Judge
    else:
        raise ValueError(f'Invalid task: {task}')

    print('Total data: ', len(df))
    print('Total Distinct Queries: ', len(df['query'].unique()))
    seen_set = set()
    # if os.path.exists(f'reasoning_by_llm/nuggets.{model_name}.{task}.all.txt'):
    #     print(f'Continue {ds} due to existing file')
    #     continue
    for i, ds_row in df.iterrows():
            query = ds_row['query']
            if query in seen_set:
                continue
            start_time = time.time()
            seen_set.add(query)
            q = ds_row['qid']
            dataset = ds_row['dataset']
            response = None
            local_message = copy.deepcopy(messages)
            local_message[1]['content'] = local_message[1]['content'].format(query, previous_list, len(previous_list))
            # print('\n--------\n',local_message,'\n---------\n')
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=local_message,
                    temperature=0,
                    top_p=1
                )
                if model_name == 'qwen3:8b':
                    data = response.choices[0].message.content.split('</think>')[-1].strip()
                else:   
                    data = response.choices[0].message.content
                previous_list = ast.literal_eval(data)
            except Exception as e:
                print(response)
                print(f"Error processing QID {q}: {e}")
                continue
            
            with open(f'reasoning_by_llm/reasons.{model_name}.{task}.all.generation.reasons.txt','a') as output:
                output.write(f'{dataset}\t{q}\t{query}\t{previous_list}\n')
            eta = (time.time() - start_time) * (len(df) - i) / 60
            print('[{}/{}] ETA:{:.2f} min : {} {} {} {} '.format(i, len(df), 
                                                        eta,
                                                        q, query, previous_list[:2],len(previous_list)))
        
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen3:8b')
    parser.add_argument('--task', type=str, default='h2g')
    args = parser.parse_args()
    print('Starting Reasoning...')
    os.makedirs('reasoning_by_llm', exist_ok=True)
    
    df_19 = pd.read_csv('datasets/test2019-queries-filterd.tsv', sep='\t', names=['qid', 'query'])
    df_20 = pd.read_csv('datasets/test2020-queries-filterd.tsv', sep='\t', names=['qid', 'query'])
    df_21 = pd.read_csv('datasets/test2021-queries-filterd.tsv', sep='\t', names=['qid', 'query'])
    df_22 = pd.read_csv('datasets/test2022-queries-filterd.tsv', sep='\t', names=['qid', 'query'])
    df_19['dataset'] = 'dl19'
    df_20['dataset'] = 'dl20'
    df_21['dataset'] = 'dl21'
    df_22['dataset'] = 'dl22'
    model_name = args.model_name
    task = args.task
    previous_list =[]   
    
    df = pd.concat([df_19, df_20, df_21, df_22])
    df = df.reset_index(drop=True)
    
    recursive_reasoning(df, task)
    print('Done!')
    
    
if __name__ == '__main__':
    main()
