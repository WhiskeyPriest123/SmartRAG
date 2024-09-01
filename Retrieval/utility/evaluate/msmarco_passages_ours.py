"""
    Evaluate MS MARCO Passages ranking.
"""

import os
import math
import tqdm
import ujson
import random
import json
from beir.retrieval.evaluation import EvaluateRetrieval


from argparse import ArgumentParser
from collections import defaultdict
from colbert.utils.utils import print_message, file_tqdm


import numpy as np


import numpy as np


def calculate_metrics_at_num(qid2positives, qid2ranking, num):
    MRR_at_num = 0
    nDCG_at_num = 0
    Recall_at_num = 0
    MAP_at_num = 0
    num_queries = len(qid2positives)

    for qid, positives in qid2positives.items():
        positives_set = set(positives)
        rankings = sorted(qid2ranking[qid], key=lambda x: x[0])[:num]
        
        # MRR@num
        mrr_temp = [1/(rank+1) for rank, pid, _ in rankings if pid in positives_set]
        MRR_at_num += max(mrr_temp) if mrr_temp else 0

        # nDCG@num
        DCG_at_num = sum([(2**rankings[i][2] - 1) / np.log2(i+2) for i in range(len(rankings))])
        ideal_rankings = sorted(rankings, key=lambda x: x[2], reverse=True)
        IDCG_at_num = sum([(2**ideal_rankings[i][2] - 1) / np.log2(i+2) for i in range(len(ideal_rankings))])
        nDCG_at_num += (DCG_at_num / IDCG_at_num) if IDCG_at_num > 0 else 0
        
        # Recall@num and MAP@num
        relevant_retrieved = [1 for _, pid, _ in rankings if pid in positives_set]
        Recall_at_num += sum(relevant_retrieved) / len(positives)
        MAP_at_num += sum(relevant_retrieved) / len(rankings) if rankings else 0

    MRR_at_num /= num_queries
    nDCG_at_num /= num_queries
    Recall_at_num /= num_queries
    MAP_at_num /= num_queries

    return {'MRR':MRR_at_num, "nDCG":nDCG_at_num, "Recall":Recall_at_num, "MAP": MAP_at_num,"N":num}






# def compute_metrics(qid2positives, qid2ranking):
#     MRR_scores = []
#     nDCG_scores = []
#     recall_scores = []
#     AP_scores = []
    
#     for qid, positives in qid2positives.items():
#         if qid not in qid2ranking:
#             continue
        
#         # 根据分数score降序排序
#         ranking = sorted(qid2ranking[qid], key=lambda x: x[2], reverse=True)
        
#         # 获取排名中的文档ID
#         ranked_pids = [x[1] for x in ranking]
        
#         # 计算 MRR
#         ranks = [idx + 1 for idx, pid in enumerate(ranked_pids) if pid in positives]
#         if ranks:
#             MRR_scores.append(1 / min(ranks))
        
#         # 计算 Recall
#         found_relevant = len(set(positives) & set(ranked_pids))
#         recall_scores.append(found_relevant / len(positives))
        
#         # 计算 nDCG
#         DCG = sum((1 / np.log2(idx + 2) for idx, pid in enumerate(ranked_pids) if pid in positives))
#         IDCG = sum((1 / np.log2(i + 2) for i in range(1, len(positives) + 1)))
#         if IDCG > 0:
#             nDCG_scores.append(DCG / IDCG)
        
#         # 计算 MAP
#         precisions = [len(set(positives) & set(ranked_pids[:idx + 1])) / (idx + 1) for idx, pid in enumerate(ranked_pids) if pid in positives]
#         if precisions:
#             AP_scores.append(np.mean(precisions))
    
#     # 计算平均值
#     MRR = np.mean(MRR_scores) if MRR_scores else 0
#     nDCG = np.mean(nDCG_scores) if nDCG_scores else 0
#     Recall = np.mean(recall_scores) if recall_scores else 0
#     MAP = np.mean(AP_scores) if AP_scores else 0
    
#     return {
#         "MRR": MRR,
#         "nDCG": nDCG,
#         "Recall": Recall,
#         "MAP": MAP
#     }

def main(args):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)
    qid2mrr = {}
    qid2recall = {depth: {} for depth in [50, 200, 1000, 5000, 10000]}

    with open(args.qrels) as f:
        print_message(f"#> Loading QRELs from {args.qrels} ..")
        for line in file_tqdm(f):
            
            # qid, _, pid, label = map(int, line.strip().split())
            # patch here
            line = json.loads(line)
            qid = int(line[0])
            pid = int(line[1])
            label = 1
            # patch here
            
            assert label == 1

            qid2positives[qid].append(pid)

    with open(args.ranking) as f:
        print_message(f"#> Loading ranked lists from {args.ranking} ..")
        for line in file_tqdm(f):
            qid, pid, rank, *score = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)

            if len(score) > 0:
                assert len(score) == 1
                score = float(score[0])
            else:
                score = None

            qid2ranking[qid].append((rank, pid, score))

    assert set.issubset(set(qid2ranking.keys()), set(qid2positives.keys()))
    
    print(calculate_metrics_at_num(qid2positives,qid2ranking,200))
    
    
    

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)
    
    
    
    
    

    # if num_judged_queries != num_ranked_queries:
    #     print()
    #     print_message("#> [WARNING] num_judged_queries != num_ranked_queries")
    #     print_message(f"#> {num_judged_queries} != {num_ranked_queries}")
    #     print()

    # print_message(f"#> Computing MRR@10 for {num_judged_queries} queries.")

    # for qid in tqdm.tqdm(qid2positives):
    #     ranking = qid2ranking[qid]
    #     positives = qid2positives[qid]

    #     for rank, (_, pid, _) in enumerate(ranking):
    #         rank = rank + 1  # 1-indexed

    #         if pid in positives:
    #             if rank <= 10:
    #                 qid2mrr[qid] = 1.0 / rank
    #             break

    #     for rank, (_, pid, _) in enumerate(ranking):
    #         rank = rank + 1  # 1-indexed

    #         if pid in positives:
    #             for depth in qid2recall:
    #                 if rank <= depth:
    #                     qid2recall[depth][qid] = qid2recall[depth].get(qid, 0) + 1.0 / len(positives)

    # assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)

    # print()
    # mrr_10_sum = sum(qid2mrr.values())
    # print_message(f"#> MRR@10 = {mrr_10_sum / num_judged_queries}")
    # print_message(f"#> MRR@10 (only for ranked queries) = {mrr_10_sum / num_ranked_queries}")
    # print()

    # for depth in qid2recall:
    #     assert len(qid2recall[depth]) <= num_ranked_queries, (len(qid2recall[depth]), num_ranked_queries)

    #     print()
    #     metric_sum = sum(qid2recall[depth].values())
    #     print_message(f"#> Recall@{depth} = {metric_sum / num_judged_queries}")
    #     print_message(f"#> Recall@{depth} (only for ranked queries) = {metric_sum / num_ranked_queries}")
    #     print()

    # if args.annotate:
    #     print_message(f"#> Writing annotations to {args.output} ..")

    #     with open(args.output, 'w') as f:
    #         for qid in tqdm.tqdm(qid2positives):
    #             ranking = qid2ranking[qid]
    #             positives = qid2positives[qid]

    #             for rank, (_, pid, score) in enumerate(ranking):
    #                 rank = rank + 1  # 1-indexed
    #                 label = int(pid in positives)

    #                 line = [qid, pid, rank, score, label]
    #                 line = [x for x in line if x is not None]
    #                 line = '\t'.join(map(str, line)) + '\n'
    #                 f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--annotate', dest='annotate', default=False, action='store_true')

    args = parser.parse_args()

    if args.annotate:
        args.output = f'{args.ranking}.annotated'
        assert not os.path.exists(args.output), args.output

    main(args)
