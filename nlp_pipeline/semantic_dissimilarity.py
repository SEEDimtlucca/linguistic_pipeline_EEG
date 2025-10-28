import os
import logging
import string
import torch
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def calculate_semantic_dissimilarity(filepath, output_dir):
    """
    Calculates word-level semantic dissimilarity for a text file using the UmBERTo Italian language model.

    Semantic dissimilarity measures how semantically "unexpected" a word is given its preceding context.
    It is computed as 1 - cosine similarity between the embedding of the current token and the mean 
    embedding of a preceding window of tokens (default 20 tokens). High values indicate semantic divergence,
    low values indicate coherence with context.

    For each input file, a dedicated subfolder is created (named after the file) where a CSV file 
    with word-level dissimilarity values is saved.

    Parameters:
        filepath (str): Path to the input .txt file.
        output_dir (str): Root directory where output subfolders will be created.

    Outputs:
        - CSV file containing word-level semantic dissimilarity:
          columns include the word and its semantic dissimilarity value.

    Notes:
        - Apostrophes and variant unicode characters are normalized to standard ASCII apostrophe.
        - Tokens are aggregated into words, handling subword tokens and apostrophes correctly.
        - Special tokens from the tokenizer (CLS, SEP, PAD) are ignored.
        - Debug information prints replaced apostrophes and quotes.
    """

    logging.info(f"Processing file: {filepath}")

    model_name = "Musixmatch/umberto-commoncrawl-cased-v1"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ").replace("\r", " ")

    apostrofi_vari = ["’", "‘", "ʼ", "＇", "‛", "´", "`", "ʹ", "ʽ", "ʾ", "ʿ", "ˈ", "ˊ", "ˋ", "âĢĻ"]
    for a in apostrofi_vari:
        text = text.replace(a, "'")
    
    text = re.sub(r"'(?=[A-Za-zÀ-ÖØ-öø-ÿ])", "' ", text)

    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    max_len = 512
    stride = 256
    chunks = [input_ids[i:i+max_len] for i in range(0, len(input_ids), stride)]

    all_tokens = []
    all_dissimilarities = []
    all_tokens = []

    for i, chunk in enumerate(chunks):
        inputs = {"input_ids": torch.tensor([chunk])}
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[-1][0]

        embeddings = hidden_states.cpu().numpy()
        chunk_tokens = tokenizer.convert_ids_to_tokens(chunk)

        dissimilarities = []
        for t in range(len(embeddings)):
            start = max(0, t - 20)
            if start == t:
                dissimilarities.append(np.nan)
            else:
                context = np.mean(embeddings[start:t], axis=0, keepdims=True)
                sim = cosine_similarity(embeddings[t].reshape(1, -1), context)[0,0]
                dissimilarities.append(1 - sim)

        if i > 0:
            overlap = stride
            chunk_tokens = chunk_tokens[overlap:]
            dissimilarities = dissimilarities[overlap:]

        all_tokens.extend(chunk_tokens)
        all_dissimilarities.extend(dissimilarities)


    
    logging.info(f"All tokens: {len(all_tokens)}")

    special_set = set(filter(None, [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]))
    apostrofo= "âĢĻ"

    filtered_tokens = []
    filtered_dissim = []
    for tok, diss in zip(all_tokens, all_dissimilarities):
        if tok in special_set or tok is None:
            continue
        if tok == apostrofo:
            filtered_tokens.append("'")
            filtered_dissim.append(diss)
        else:
            filtered_tokens.append(tok)
            filtered_dissim.append(diss)

    words = []
    words_dissim = []
    current_tokens = []
    current_dissim = []

    for tok, diss in zip(filtered_tokens, filtered_dissim):
        if tok == "'":
            if current_tokens:
                word = tokenizer.convert_tokens_to_string(current_tokens).strip().strip('"')
                if word:
                    word = word + "'"  
                    agg = float(np.nanmean(current_dissim)) if current_dissim else np.nan
                    words.append(word)
                    words_dissim.append(agg)

            current_tokens = []
            current_dissim = []
            continue


        if tok.startswith("▁") or tok.startswith("â–"):  # nuovo inizio parola
            if current_tokens:
                word = tokenizer.convert_tokens_to_string(current_tokens).strip().strip('"')
                if word:
                    agg = float(np.nanmean(current_dissim)) if len(current_dissim) > 0 else np.nan
                    words.append(word)
                    words_dissim.append(agg)
            tok_clean = tok.lstrip("▁").lstrip("â–")
            current_tokens = [tok_clean]
            current_dissim = [diss]
        else:
            current_tokens.append(tok)
            current_dissim.append(diss)

    if current_tokens:
        word = tokenizer.convert_tokens_to_string(current_tokens).strip().strip('"')
        if word:
            agg = float(np.nanmean(current_dissim)) if len(current_dissim) > 0 else np.nan
            words.append(word)
            words_dissim.append(agg)

    extra_punct = {"“", "”", "«", "»", "„", "‟", "‹", "›"}
    punct_to_remove = (set(string.punctuation) | extra_punct) - {"'"}
    def clean_word(w):
        return w.strip("".join(punct_to_remove))

    words_cleaned = []
    dissim_cleaned = []
    for w, d in zip(words, words_dissim):
        w_clean = clean_word(w)
        if w_clean:
            words_cleaned.append(w_clean)
            dissim_cleaned.append(d)

    df = pd.DataFrame({
        "word": words_cleaned,
        "semantic_dissimilarity": dissim_cleaned
    })
   
    name_base = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join(output_dir, name_base)
    os.makedirs(file_output_dir, exist_ok=True)
    csv_path = os.path.join(file_output_dir, f"dissimilarity_{name_base}.csv")

    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")

    
