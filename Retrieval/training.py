from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer
import argparse  




if __name__=='__main__':
    parse = argparse.ArgumentParser(description='The parameters of training') 
    parse.add_argument("--experiment_name",type=str)
    parse.add_argument("--bsize",type=int)
    parse.add_argument("--max_steps",type=int)
    parse.add_argument("--root",type=str)
    parse.add_argument("--checkpoint",type=str)
    parse.add_argument("--triples_file",type=str)
    parse.add_argument("--queries_file",type=str)
    parse.add_argument("--collection_file",type=str)
    args = parse.parse_args()
    


    with Run().context(RunConfig(nranks=1, experiment=args.experiment_name)):

        config = ColBERTConfig(
            bsize=args.bsize,
            root=args.root,
            checkpoint=args.checkpoint,
            maxsteps = args.max_steps
        )
        
        trainer = Trainer(
            triples=args.triples_file,
            queries=args.queries_file,
            collection=args.collection_file,
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint=args.checkpoint)

        print(f"Saved checkpoint to {checkpoint_path}...")
        