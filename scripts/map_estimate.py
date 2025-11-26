from probabilistic_word_embeddings.embeddings import Embedding
from probabilistic_word_embeddings.estimation import map_estimate
from embedding_uncertainty.laplace_approx import gradient_tf, gradient

import numpy as np
import tensorflow as tf
from trainerlog import get_logger
LOGGER = get_logger("map_estimate")
LOGGER.info("Load modules..")
from pathlib import Path
import random, json

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--data_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save_path", type=str, default="map_estimate.pkl")
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

    LOGGER.info(f"Data length: {len(data)}")

    e_map = Embedding(vocab, dimensionality=args.dim, lambda0=args.dim)
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
    e_grad = gradient_tf(e_map, data)

    print(e_grad["word0"])
    print(e_grad["word2"])

    e_map = map_estimate(e_map, data_generator=datagen(), model="sgns", epochs=args.epochs, N=datalen, batch_size=batch_size)
    e_grad = gradient_tf(e_map, data)

    print(e_grad["word0"])
    print(e_grad["word2"])
    e_map = map_estimate(e_map, data_generator=datagen(), model="sgns", epochs=args.epochs, N=datalen, batch_size=batch_size)
    e_grad = gradient_tf(e_map, data)

    print(e_grad["word0"])
    print(e_grad["word2"])

    e_map.save(args.save_path)