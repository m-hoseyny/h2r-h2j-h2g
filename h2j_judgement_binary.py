import os
import json
import re
import openai
import ast
import argparse

import tqdm


client = openai.OpenAI(api_key='ollama', 
                           base_url='http://localhost:11434/v1')

def main(dataset, model_name):
    BASEPAHT = '/mnt/data/mohammad-hosseini/datasets'
    for year in dataset:
        print('Going to process year ', year)
        queries = {}
        for line in tqdm.tqdm(open(f'{BASEPAHT}/test20{year}-queries-filterd.tsv', 'r').readlines(), desc=f'Loading queries for year {year}'):
            qid, query = line.split('\t')
            queries[qid] = query

        text_col = {}
        for line in tqdm.tqdm(open(f'{BASEPAHT}/qrels_text.dl{year}', 'r').readlines(), desc=f'Loading texts for year {year}'):
            docid = line.split('\t')[0]
            doctext = line.replace(docid, '').replace('\t', '')
            text_col[docid] = doctext
        

        output = open(f'binary_judge/binary_{model_name}_dl{year}.txt', 'a')

        for line in tqdm.tqdm(open(f'{BASEPAHT}/qrels.dl{year}-passage.txt', 'r').readlines(), desc=f'Judging for year {year}'):
            qid, _, docid, _ = line.split()

            messages = [
                {
                    "role": "system",
                    "content": (
                        " You are an expert assessor making TREC relevance judgments. \
                        You will be given a TREC topic and a portion of a document. \
                        If any part of the document is relevant to the topic, answer “Yes”. \
                        If not, answer “No”. Remember that the TREC relevance condition states that a document is relevant to a topic \
                        if it contains information that is helpful in satisfying the user’s information need described by the topic. \
                        A document is judged relevant if it contains information that is on-topic and of potential value to the user. /no_think "
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f" Indicate if the passage is relevant for the question. \
                        Question: {queries[qid]}\n \
                        Passage: {text_col[docid]} \n \
                        Relevant?"
                    )
                }
            ]

            try:
                response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0)
            except Exception as e:
                # print(e)
                continue
            rate = response.choices[0].message.content
            if model_name == 'qwen3:8b':
                rate = str(rate).split('</think>')[-1]
            if 'yes' in rate.lower():
                rate = 1
            else:
                rate = 0
            # print('binary',model_name,year, qid, docid, rate)
            output.write(f"binary_judge/h2j_binary_{model_name}_{year}_{qid} 0 {docid} {rate}\n")
    output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=list, default=['20','21', '22','19'], help='name the collections')
    parser.add_argument('--model_name', type=str, default='qwen3:8b', help='Model name to use')
    os.makedirs('binary_judge', exist_ok=True)
    args = parser.parse_args()

    main(dataset=args.dataset, model_name=args.model_name)