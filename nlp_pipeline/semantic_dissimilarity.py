from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def calculate_semantic_dissimilarity(filepath, output_dir):
    logging.info(f"Processing file: {filepath}")

    model_name = "Musixmatch/umberto-commoncrawl-cased-v1"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ").replace("\r", " ")

    words = tokenizer.tokenize(text)

    embeddings = []
    for i, word in enumerate(words):
        inputs = tokenizer(words[:i+1], is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # embedding della parola corrente (ultimo token dell'ultimo layer)
        v_t = outputs.hidden_states[-1][0, -1, :].numpy()
        embeddings.append(v_t)


    # calcola dissimilarità rispetto al contesto precedente
    dissimilarities = []
    window_size = 20
    for t in range(len(embeddings)):
        if t == 0:
            dissimilarities.append(np.nan)
        else:
            start = max(0, t - window_size)
            context = np.mean(embeddings[start:t], axis=0, keepdims=True)
            v_t = embeddings[t].reshape(1, -1)
            sim = cosine_similarity(v_t, context)[0, 0]
            dissimilarities.append(1 - sim)

    df = pd.DataFrame({"word": words, "semantic_dissimilarity": dissimilarities})
    df.to_csv(f"{output_dir}/dissimilarity.csv", index=False)


calculate_semantic_dissimilarity("data\\texts_03\\01_03.txt", "output")

df = pd.read_csv("output\\dissimilarity.csv")

plt.plot(df["semantic_dissimilarity"])
plt.show()