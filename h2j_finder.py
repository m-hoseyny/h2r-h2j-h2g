import pandas as pd
import argparse


def hard_to_judge_finder(qrel_file_path, hard_to_judge_path, quantile):
    qrels_dl21 = pd.read_csv(qrel_file_path, sep=" ", header=None, names=["query", "dummy", "docid", "actual_relevance"])
    run_qrels_umbrela = pd.read_csv(hard_to_judge_path, sep=" ", header=None, names=["query", "dummy", "docid", "relevance"])
    
    umbrela_judge = run_qrels_umbrela.merge(qrels_dl21, on=["query", "docid"], how="left")
    # print(umbrela_judge.head())
    umbrela_judge['diff'] = abs(umbrela_judge['relevance'] - umbrela_judge['actual_relevance'])

    umbrela_judge = umbrela_judge[umbrela_judge['diff'] > 2]
    distribution_judge_umbrela = umbrela_judge.groupby('query').count().sort_values(['diff'], ascending=True).reset_index()
    quantile_judge_umbrela = distribution_judge_umbrela['diff'].quantile(1 - quantile)
    hard_to_judge_umbrela = distribution_judge_umbrela[distribution_judge_umbrela['diff'] > quantile_judge_umbrela]
    return hard_to_judge_umbrela

def hard_to_judge_binary_finder(qrel_file_path, hard_to_judge_path, quantile):
    qrels_dl21 = pd.read_csv(qrel_file_path, sep=" ", header=None, names=["query", "dummy", "docid", "actual_relevance"])
    run_qrels_umbrela = pd.read_csv(hard_to_judge_path, sep=" ", header=None, names=["query", "dummy", "docid", "relevance"])

    umbrela_judge = run_qrels_umbrela.merge(qrels_dl21, on=["query", "docid"], how="left")
    umbrela_judge['actual_relevance'] = umbrela_judge['actual_relevance'].apply(lambda x: 1 if int(x) > 1 else 0)
    umbrela_judge['diff'] = abs(umbrela_judge['relevance'] - umbrela_judge['actual_relevance'])
    distribution_judge_umbrela = umbrela_judge.groupby('query').sum().sort_values(['diff'], ascending=True).reset_index()
    quantile_judge_umbrela = distribution_judge_umbrela['diff'].quantile(1 - quantile)
    hard_to_judge_umbrela = distribution_judge_umbrela[distribution_judge_umbrela['diff'] > quantile_judge_umbrela]
    return hard_to_judge_umbrela

hard_to_judge_umbrela_paths = [
    'paper_repo/modified_qrels/qrels.dl19-passage_llama3.2:latest_0123_0_1.txt',
    'paper_repo/modified_qrels/qrels.dl20-passage_llama3.2:latest_0123_0_1.txt',
    'paper_repo/modified_qrels/qrels.dl21-passage_llama3.2:latest_0123_0_1.txt',
    'paper_repo/modified_qrels/qrels.dl22-passage_llama3.2:latest_0123_0_1.txt',
]

hard_to_judge_binary_paths = [
    'paper_repo/binary_judge/binary_llama3.2:latest_dl19.txt',
    'paper_repo/binary_judge/binary_llama3.2:latest_dl20.txt',
    'paper_repo/binary_judge/binary_llama3.2:latest_dl21.txt',
    'paper_repo/binary_judge/binary_llama3.2:latest_dl22.txt',
]

qrels_path = [
    'datasets/datasets/qrels.dl19-passage.txt',
    'datasets/datasets/qrels.dl20-passage.txt',
    'datasets/datasets/qrels.dl21-passage.txt',
    'datasets/datasets/qrels.dl22-passage.txt',
]

hard_to_judge_ubmrela = []
for i in range(len(qrels_path)):
    data = hard_to_judge_finder(qrels_path[i], hard_to_judge_umbrela_paths[i], 0.3)
    print('Hard to judge dataset umbrela: {}'.format(data))
    hard_to_judge_ubmrela.append(data)
    
hard_to_judge_ubmrela = pd.concat(hard_to_judge_ubmrela)
hard_to_judge_ubmrela.to_csv('hard_to_judge_ubmrela.csv')

hard_to_judge_binary = []
for i in range(len(qrels_path)):
    data = hard_to_judge_binary_finder(qrels_path[i], hard_to_judge_binary_paths[i], 0.3)
    print('Hard to judge dataset binary: {}'.format(data))
    hard_to_judge_binary.append(data)
    
hard_to_judge_binary = pd.concat(hard_to_judge_binary)
hard_to_judge_binary.to_csv('hard_to_judge_binary.csv')
