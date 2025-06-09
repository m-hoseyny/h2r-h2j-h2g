from openai import OpenAI
import tqdm
import pandas as pd
import json
import argparse
import os


# Initialize client
client = OpenAI(
    # base_url='http://192.168.1.8:10123/v1',
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

smsg = """
You are a highly experienced and accurate assessor for TREC.
/no_think
"""

def answer(question, model="llama3.3:latest"):
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": smsg},
        {"role": "user", "content": question},
      ]
    )
    return completion.choices[0].message.content.lower().replace("\n", " ")

def generate_prompt(question, passage_a, passage_b):
    return """
Select the passage that answers the question better. Just answer 1 or 2, without any explanation or extra verbiage.  If both passages are similar, select the simplest and clearest.
Question:
{question}
Passage 1:
{passage_a}
Passage 2:
{passage_b}
""".format(question=question, passage_a=passage_a, passage_b=passage_b)

def judge(question, passage_a, passage_b, model="llama3.3:latest"):
    prompt = generate_prompt(question, passage_a, passage_b)
    response = answer(prompt, model=model)
    if "Passage 1" in response:
        return 1
    if "Passage 2" in response:
        return -1 
    if "1" in response and "2" not in response:
        return 1
    if "1" not in response and "2" in response:
        return -1
    return 0

def pref(question, passage_a, passage_b, model="llama3.3:latest"):
    a = judge(question, passage_a, passage_b, model=model)
    b = judge(question, passage_b, passage_a, model=model)
    if a == 1 and b == -1:
      return 1
    if a == -1 and b == 1:
      return -1
    if a == 0:
      return -b
    if b == 0:
      return a
    return 0

def get_passage_msv2(pid):
    (string1, string2, bundlenum, position) = pid.split("_")
    assert string1 == "msmarco" and string2 == "passage"

    with open(
        f"msmarco_v2_passage/msmarco_passage_{bundlenum}", "rt", encoding="utf8"
    ) as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document["pid"] == pid
        return document["passage"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate BERTScore between two qrels files')
    parser.add_argument('-q', '--qrels', help='Path to reference qrels file')
    parser.add_argument('-g', '--gp', help='Generated Passage Json file') # gp: Generated Passage
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-m', '--model', help='Model to use for pairwise judge', default='llama3.3:latest')
    args = parser.parse_args()
    
    print('Loading files...')
    results = pd.read_json(args.gp, lines=True)
    qrels = pd.read_csv(args.qrels, 
                        sep=' ', header=None, names=['query_id', 'dummy', 'pid', 'score'])
    print('Files loaded')
    
    pref_scores = []
    print('Start calculating pairwise judge')
    previous_score_df_qids = set()
    if os.path.exists(args.output):
        previous_score_df = pd.read_json(args.output, lines=True)
        previous_score_df_qids = set(previous_score_df['query_id'])
        pref_scores = previous_score_df.to_dict('records')
    for idx, result in tqdm.tqdm(results.iterrows(), 
                                 desc="Calculating Pairwise Judge",
                                 total=len(results)):
        qid = result['query_id']
        if qid in previous_score_df_qids:
            continue
        for idx, qrel in tqdm.tqdm(qrels[qrels['query_id'] == qid].iterrows(),
                                   desc=f"Query {qid}",
                                   total=len(qrels[qrels['query_id'] == qid])):
            pid = qrel['pid']
            passage = get_passage_msv2(pid)
            
            pref_score = pref(result['query'], passage, result['response'], model=args.model)
            pref_scores.append({
                'query_id': qid,
                'main_performance': passage,
                'generated':  result['response'],
                'pid': pid,
                'pref_score': pref_score
            })
    
        pref_scores_df = pd.DataFrame(pref_scores)
        pref_scores_df.to_json(args.output, orient='records', lines=True)


