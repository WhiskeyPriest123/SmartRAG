from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import argparse


if __name__=='__main__':
    parse = argparse.ArgumentParser(description='The parameters of retrieving') 
    parse.add_argument("--experiment_name",type=str)
    parse.add_argument("--save_file",type=int)
    parse.add_argument("--depth",type=int)
    parse.add_argument("--root",type=str)
    parse.add_argument("--checkpoint",type=str)
    parse.add_argument("--queries_file",type=str)
    parse.add_argument("--index_name",type=str)
    parse.add_argument("--n_gpu",type=int)

    args = parse.parse_args()
    
    
    with Run().context(RunConfig(nranks=args.n_gpu, experiment=args.experiment_name)):
        config = ColBERTConfig(
            root=args.root,
            checkpoint=args.checkpoint
            )
        searcher = Searcher(index=args.index_name, config=config)
        queries = Queries(args.queries_file)
        ranking = searcher.search_all(queries, k=args.depth)
        ranking.save(args.save_file)
        