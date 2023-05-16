from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.preprocessing import preprocess_standard
from probabilistic_word_embeddings.estimation import map_estimate
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import progressbar
from nltk.stem import PorterStemmer
from randomread import sample as sample_from_file
PS = PorterStemmer()

def main(args):
    with open(args.vocabpath) as f:
        topwords = json.load(f)
        
    millions = int(args.N / 1e6)
    for r in range(args.runs):
        print(f"ROUND {r}")
        print("True estimator")
        text = sample_from_file(args.datapath, args.N).lower().split()
        text = [PS.stem(wd) for wd in progressbar.progressbar(text)]
        text = [wd if wd in topwords else "oov" for wd in progressbar.progressbar(text)]
        # Train an embedding on the resampled data
        #vocabulary = set(text)
        text, vocabulary = preprocess_standard(text)
        vocabulary = topwords

        e = Embedding(vocabulary, args.dim)
        e = map_estimate(e, text, model="cbow", ws=args.ws, batch_size=args.batch_size, epochs=args.epochs, training_loss=True)
        
        # Save embedding
        trained_model_folder = Path(args.outfolder)
        resample_path = trained_model_folder / "true-ci-vs-bootstrap" / f"ref-{millions}M-dim-{args.dim}-ws-{args.ws}-no-{r}.pkl"
        e.save(resample_path.absolute())
        
        resample_len = args.resample_len
        datapoints = len(text) // resample_len
        datapoints = list(range(datapoints))

        for bs_r in range(args.bs_runs):
            print("Bootstrap")
            r_indices = np.random.choice(datapoints, size=len(datapoints))
            # Resampled data
            text_r = []
            for ix in r_indices:
                start, end = ix * resample_len, (ix + 1) * resample_len
                text_r += text[start:end]

            e = Embedding(vocabulary, args.dim)
            e = map_estimate(e, text_r, model="cbow", ws=args.ws, batch_size=args.batch_size, epochs=args.epochs, training_loss=True)

            resample_path = trained_model_folder / "true-ci-vs-bootstrap" / f"bs-{millions}M-dim-{args.dim}-ws-{args.ws}-no-{r}-bsno-{bs_r}.pkl"
            e.save(resample_path.absolute())

        
if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--datapath", type=str)
    argparser.add_argument("--N", type=int, required=True)
    argparser.add_argument("--vocabpath", type=str, default="top1k-stemmed.json")
    argparser.add_argument("--dim", type=int, default=100)
    argparser.add_argument("--model", type=str, default="cbow")
    argparser.add_argument("--ws", type=int, default=2)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--epochs", type=int, default=15)
    argparser.add_argument("--resample_len", type=int, default=5000)
    argparser.add_argument("--runs", type=int, default=10)
    argparser.add_argument("--bs_runs", type=int, default=10)
    argparser.add_argument("--outfolder", type=str, default="trained")
    args = argparser.parse_args()
    print(args)
    main(args)