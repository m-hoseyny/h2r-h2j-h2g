
import pandas as pd
import numpy as np
from trectools import TrecRun, TrecQrel, TrecEval
import argparse


def hard_to_retrieve_finder(run_file_path, qrel_file_path, quantile):
    run_dl21 = TrecRun(run_file_path)
    qrels_dl21 = TrecQrel(qrel_file_path)
    # Filter run to only include queries present in qrels
    valid_qids = qrels_dl21.qrels_data["query"].unique()
    filtered_run_df = run_dl21.run_data[run_dl21.run_data["query"].isin(valid_qids)]
    # Update the run object
    run_dl21.run_data = filtered_run_df
    # Now evaluate
    evaluator = TrecEval(run_dl21, qrels=qrels_dl21)
    ndcg_per_query_bm25 = evaluator.get_ndcg(10, per_query=True, removeUnjudged=True)

    lower_band_bm25 = ndcg_per_query_bm25.sort_values(['NDCG@10'], ascending=True)['NDCG@10'].quantile(quantile)
    hard_ndcg_per_query_bm25 = ndcg_per_query_bm25[ndcg_per_query_bm25['NDCG@10'] < lower_band_bm25]
    hard_ndcg_per_query_bm25 = hard_ndcg_per_query_bm25.reset_index()
    return hard_ndcg_per_query_bm25

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_file_path', required=True, help='Path to the run file')
    parser.add_argument('--qrel_file_path', required=True, help='Path to the qrel file')
    parser.add_argument('--quantile', type=float, default=0.3, help='Quantile to use for hard to retrieve queries')
    parser.add_argument('--output', required=True, help='Path to the output file')
    args = parser.parse_args()
    hard_ndcg_per_query_bm25 = hard_to_retrieve_finder(args.run_file_path, args.qrel_file_path, args.quantile)
    hard_ndcg_per_query_bm25.to_csv(args.output, index=False)
    print(hard_ndcg_per_query_bm25)
