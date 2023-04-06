from pathlib import Path
import random

def sample_from_textfile(path, no_words, chunk_size=50000, word_len_estimate=10):
    texts = []
    filepath = Path(path).expanduser()
    file_stats = filepath.stat()
    file_len_bytes = file_stats.st_size
    print(file_len_bytes)
    #with open(path, "rb") as f:
    
    chunks = int(no_words * word_len_estimate / chunk_size)
    print(f"Generate {chunks} chunks")
    
    chunks = [random.randint(0, file_len_bytes-chunk_size) for _ in range(chunks)]
    chunks = sorted(chunks)
    print(chunks)

    f = filepath.open("rb")    
    for chunk in chunks:
        current_pos = f.tell()
        print(f"Current position in file {current_pos}")
        f.seek(chunk - current_pos)
        print(f"Go to {chunk}. Current position in file {f.tell()}")
        current_bytes = f.read(chunk_size)
        current_text = current_bytes.decode("utf-8")
        current_text = current_text.split()
        current_text = current_text[1:]
        current_text = current_text[:-1]
        current_text = " ".join(current_text)
        texts.append(current_text)
        print(current_text[:10])
    f.close()
        
    random.shuffle(texts)
    current_no_words = 0
    text = ""
    for t in texts:
        new_no_words = len(t.split())
        if current_no_words + new_no_words > no_words:

            t_split = t.split()
            no_words - current_no_words
        elif current_no_words < no_words:
            current_no_words += new_no_words
            text += " " + " ".join(t.split())
        else:
            break
    return text