from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.estimation import map_estimate
from probabilistic_word_embeddings.models import sgns_likelihood
from probabilistic_word_embeddings.evaluation import posterior_mean, nearest_neighbors
from probabilistic_word_embeddings.evaluation import evaluate_word_similarity
import numpy as np
import tensorflow as tf
from trainerlog import get_logger
LOGGER = get_logger("gibbs")
LOGGER.info("Load modules..")
from pathlib import Path
import random, json

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--sample_folder", type=str, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lambda0", type=float, default=None)
    args = parser.parse_args()
    LOGGER.train(f"Args: {args}")

    data, vocab = [], set()
    if args.train_data is not None:
        with open(args.train_data, "rb") as f:
            d = json.load(f)
        LOGGER.debug(f"Keys: {d.keys()}")
        random.shuffle(d["data"])
        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            vocab.add(w)
            vocab.add(elem["w"])
            data.append((w,v,x))

    if args.data_len is not None:
        data = data[:args.data_len]

    sample_folder = Path(args.sample_folder)
    ref_emb = Embedding(saved_model_path=str(list(sample_folder.glob("*.pkl"))[-1].absolute()))
    lambda0 = args.lambda0
    if args.lambda0 is None:
        lambda0 = ref_emb.lambda0
    dimensionality = ref_emb.dimensionality
    
    e_map = Embedding(vocab, dimensionality=dimensionality, lambda0=lambda0)
    datalen = len(data)
    batch_size = args.batch_size
    def datagen():
        while True:
            i, j, x = [], [], []
            for _ in range(batch_size):
                elem = data[random.randint(0, datalen-1)]
                w, v, x_i = elem
                i.append(w)
                j.append(v)
                x.append(x_i)
            
            yield tf.constant(i), tf.constant(j), tf.constant(x, dtype=tf.float64)
        
    e_map = map_estimate(e_map, data_generator=datagen(), model="sgns", epochs=args.epochs, N=datalen, batch_size=batch_size)
    e_map.save(f"map-K-{args.dim}-N-{args.data_len}-{args.epochs}.pkl")
    
    # Discard warmup samples
    samples = sorted(sample_folder.glob("*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
    # By default the first half
    if args.warmup is None:
        samples = samples[len(samples) // 2:]
    
    e_post_mean = posterior_mean([str(s.absolute()) for s in samples])
    
    # Load in test dataset
    test_i = []
    test_j = []
    test_x = []
    with open(args.test_data, "rb") as f:
        d = json.load(f)
        for elem in d["data"]:
            w, v, x = elem["v"], elem["w"] + "_c", elem["x"]
            test_i.append(w)
            test_j.append(v)
            test_x.append(x)
    test_i = tf.constant(test_i)
    test_j = tf.constant(test_j)
    test_x = tf.constant(test_x, dtype=tf.float64)
    
    ll_map = sgns_likelihood(e_map, test_i, test_j, test_x)
    ll_pm = sgns_likelihood(e_post_mean, test_i, test_j, test_x)
    
    print(ll_map)
    print(ll_pm)
    print(tf.reduce_max(ll_pm))
    print(tf.reduce_min(ll_pm))
    print("MAP mean", np.mean(ll_map))
    print("PM mean", np.mean(ll_pm))
    
    df_map = evaluate_word_similarity(e_map)
    print("MAP wordsim")
    print(df_map)
    df_map_mean = np.mean(df_map["Rank Correlation"])
    print(df_map_mean)
    df_pm = evaluate_word_similarity(e_post_mean)
    print("PM wordsim")
    print(df_pm)
    df_pm_mean = np.mean(df_pm["Rank Correlation"])
    print(df_pm_mean)
    #print(nearest_neighbors(e_post_mean, ["StarWars1977", "MissionImpossible1996", "Jaws1975", "Titanic1997", "SpaceJam1996", "Pinocchio1940", "GodfatherThe1972"], K=5))
