from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
import tensorflow as tf
import numpy as np
from pathlib import Path

def main(args):
    resamples = 10
    resample_len = 500
    dim = args.dim
    epochs = args.epochs
    text = open(args.datapath).read().lower().split()
    print(text[:10])
    #text, vocabulary = preprocess_standard(text)
    vocabulary = set(text)
    text_val = None
    if args.datapath_val is not None:
        text_val = open(args.datapath_val).read().lower().split()

    print(f"Train on a text of length {len(text)} with a vocabulary size of {len(vocabulary)}")

    trained_model_folder = Path("trained")
    if not trained_model_folder.exists():
        trained_model_folder.mkdir()
        
    for r in range(10):        
        # Train an embedding on the resampled data
        e = Embedding(vocabulary, dim)
        e = map_estimate(e, text, model=args.model, ws=args.ws, epochs=epochs, batch_size=args.batch_size, evaluate=False, valid_data=text_val)
        
        testwords = list(vocabulary)[:3]
        print(testwords)
        e_testwords = e[testwords]
        print(tf.tensordot(e_testwords, e_testwords, axes=[1,1]))
        # Save embedding
        resample_path = trained_model_folder / f"reference-dim-{args.dim}-ws-{args.ws}-no-{r}.pkl"
        e.save(resample_path.absolute())

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--datapath", type=str)
    argparser.add_argument("--datapath_val", type=str)
    argparser.add_argument("--dim", type=int, default=25)
    argparser.add_argument("--model", type=str, default="sgns")
    argparser.add_argument("--ws", type=int, default=2)
    argparser.add_argument("--batch_size", type=int, default=1000)
    argparser.add_argument("--epochs", type=int, default=15)
    args = argparser.parse_args()

    main(args)