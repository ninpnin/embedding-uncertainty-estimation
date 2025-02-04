from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import progressbar

def main(args):
    with open(args.vocabpath) as f:
        topwords = json.load(f)
    
    text = open(args.datapath).read().lower().split()
    text = [wd if wd in topwords else "oov" for wd in progressbar.progressbar(text)]
    #text, vocabulary = preprocess_standard(text)
    print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")
        
    for r in range(args.runs):        
        # Train an embedding on the resampled data
        e = Embedding(vocabulary, args.dim)
        e = map_estimate(e, text, model="cbow", ws=args.ws, batch_size=args.batch_size, epochs=args.epochs, training_loss=True)
        
        # Save embedding
        trained_model_folder = Path(args.outfolder)
        resample_path = trained_model_folder / f"ref-stemmedwiki-dim-{args.dim}-ws-{args.ws}-no-{r}.pkl"
        e.save(resample_path.absolute())
        
if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--datapath", type=str)
    argparser.add_argument("--vocabpath", type=str, default="top1k-stemmed.json")
    argparser.add_argument("--dim", type=int, default=100)
    argparser.add_argument("--model", type=str, default="cbow")
    argparser.add_argument("--ws", type=int, default=2)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--epochs", type=int, default=15)
    argparser.add_argument("--runs", type=int, default=10)
    argparser.add_argument("--outfolder", type=str, default="trained")
    args = argparser.parse_args()
    print(args)
    main(args)