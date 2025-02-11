from pathlib import Path
import polars as pl
import progressbar
import re
from nltk.stem.snowball import SnowballStemmer

regex = re.compile('[^a-zA-Z ]')
folder = Path("hein-daily")

LEMMATIZE = True
stemmer = SnowballStemmer("english", ignore_stopwords=True)

CHUNK_SIZE = 10000

def clean_and_lemmatize(text):
    text = text.lower()
    text = regex.sub("", text)
    text = " ".join([stemmer.stem(wd) for wd in text.split()])
    return text

for p in folder.glob("speeches_*.txt"):
    df = pl.read_csv(p, encoding="latin1", separator="|", truncate_ragged_lines=True, infer_schema_length=100000)
    no_chunks = len(df) // CHUNK_SIZE
    if len(df) % CHUNK_SIZE != 0:
        no_chunks += 1
        
    for ch_ix in range(no_chunks):
        df_ix = df.slice(ch_ix * CHUNK_SIZE, CHUNK_SIZE)
        cleaned_speeches = [clean_and_lemmatize(t) for t in progressbar.progressbar(df_ix["speech"])]
        d = {"speech_id": list(df_ix["speech_id"]), "speech": cleaned_speeches}
        df_ix = pl.DataFrame(d)
        print(df_ix)
        
        start_ix = df_ix["speech_id"].min()
        end_ix = df_ix["speech_id"].max()
        
        df_ix.write_ndjson(f"raw-speeches/stemmed-{start_ix}-{end_ix}.ndjson")
