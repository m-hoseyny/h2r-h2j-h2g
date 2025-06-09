import os
from evaluate import load
import json
import argparse
import pandas as pd
import tqdm
import pickle

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

def load_passages_msv1():
    passage_v1 = {}
    with open('/mnt/data/mohammad-hosseini/datasets/collection.tsv') as f:
        for line in tqdm.tqdm(f, desc="Loading passages v1"):
            did, passage = line.strip().split('\t')
            passage_v1[did] = passage
    return passage_v1



def load_qrels(file_path):
    """Load qrels file and return a dictionary of {qid: text}"""
    qrels_dict = {}
    unique_dids = set()
    with open(file_path, 'r') as f:
        for line in tqdm.tqdm(f, desc="Loading qrels"):
            qid, _, docid, rel = line.strip().split()
            if int(rel) > 1:  # Only consider relevant and top passages
                qrels_dict[qid] = qrels_dict.get(qid, []) + [docid]
                unique_dids.add(docid)
    return qrels_dict, unique_dids

def calculate_bert_score(qrels, ref_passages, gp, output):
    """
    Calculates BERTScore between the qrels and Generated Passage(gp) provided in the given files.
    """
    bertscore = load("bertscore")

    score_predictions = []
    previous_scores = None
    if os.path.exists(output):
        previous_scores = pd.read_json(output, orient='records', lines=True)
        print('Previous qids: ', list(previous_scores['qid'].values))
    for qid, dids in tqdm.tqdm(qrels.items(), desc="Calculating BERTScore"):
        score_prediction = {}
        if previous_scores is not None and int(qid) in list(previous_scores['qid'].astype(int).values):
            score_prediction = previous_scores[previous_scores['qid'] == int(qid)].iloc[0].to_dict()
            score_predictions.append(score_prediction)
            continue
        generated_passages = list(gp[gp['query_id'] == int(qid)]['response'].values)
        if not generated_passages:
            score_prediction['qid'] = qid
            score_prediction['scores'] = None
            score_prediction['main_performance'] = None
            score_prediction['dids'] = None
            score_predictions.append(score_prediction)
            continue
        generated_passages = generated_passages[0]
        references = [ref_passages[did] for did in dids]
        predictions = [generated_passages for i in range(len(dids))]
        score_prediction['qid'] = qid
        score_prediction['scores'] = bertscore.compute(predictions=predictions, references=references, lang="en", verbose=False)
        score_prediction['main_performance'] = generated_passages
        score_prediction['dids'] = dids
        score_predictions.append(score_prediction)

        df = pd.DataFrame(score_predictions)
        df.to_json(output, orient='records', lines=True)
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate BERTScore between two qrels files')
    parser.add_argument('-q', '--qrels', help='Path to reference qrels file')
    parser.add_argument('-g', '--gp', help='Generated Passage Json file') # gp: Generated Passage
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-p', '--passages_qrels_path', help='Passages qrels path')
    args = parser.parse_args()
    passages_qrels_path = args.passages_qrels_path
    # Load qrels files
    ref_qrels, ref_dids = load_qrels(args.qrels)
    ref_passages = {}
    passage_v1 = {}
    if 'dl19' in args.qrels or 'dl20' in args.qrels:
        passage_v1 = load_passages_msv1()
    
    if os.path.exists(passages_qrels_path):
        print('[oooo] Loading passages for qrels from pickle file')
        with open(passages_qrels_path, 'rb') as f:
            ref_passages = pickle.load(f)
    else:
        print('[xxxx] Loading passages for qrels from msv2')
        for did in tqdm.tqdm(ref_dids, desc="Loading passages for qrels"):
            if 'dl19' in args.qrels or 'dl20' in args.qrels:
                ref_passages[did] = passage_v1[did]
            else:
                ref_passages[did] = get_passage_msv2(did)
        
        with open(passages_qrels_path, 'wb') as f:
            pickle.dump(ref_passages, f)
            
    generated_passages = pd.read_json(args.gp, lines=True)
    generated_passages['query_id'] = generated_passages['query_id'].astype(int)

    calculate_bert_score(ref_qrels, ref_passages, generated_passages, args.output)