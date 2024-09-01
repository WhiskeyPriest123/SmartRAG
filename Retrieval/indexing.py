from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
import argparse


if __name__=='__main__':
    parse = argparse.ArgumentParser(description='The parameters of indexing') 
    parse.add_argument("--experiment_name",type=str)
    parse.add_argument("--root",type=str)
    parse.add_argument("--checkpoint",type=str)
    parse.add_argument("--index_name",type=str)
    parse.add_argument("--collection_file",type=str)

    args = parse.parse_args()
    
    
    with Run().context(RunConfig(nranks=1, experiment=args.experiment_name)):

        config = ColBERTConfig(
            nbits=2,
            root=args.root
        )
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.index_name,collection=args.collection_file)
        
