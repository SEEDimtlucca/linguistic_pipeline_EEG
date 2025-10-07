import os
import logging
import json
import math
import string
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import stanza
from nltk import bigrams

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

stanza.download("it")
nlp = stanza.Pipeline("it", processors="tokenize,mwt,pos,lemma,depparse", use_gpu=False)

def process_text_file(filepath, output_dir):
    """
    Processes a single text file using the Stanza NLP pipeline and extracts linguistic features.

    For each input file, a dedicated subfolder is created (named after the file, e.g. '01_03') 
    where all outputs are saved.

    Parameters:
        filepath (str): Path to the input .txt file.
        output_dir (str): Root directory where output subfolders will be created.

    Outputs:
        - A CSV file containing token-level linguistic features:
          sentence ID, token, lemma, PoS, dependency relation, head, constituency (if available),
          cleaned token/lemma, AoA (age of acquisition), and SUBTLEX-IT frequency.
        - A PNG bar chart showing the top 20 most frequent lemmas.

    Notes:
        - Output filenames are automatically derived from the input filename.
        - Constituency parsing is included if available in the sentence object.
        - AoA and frequency values are mapped from external normative datasets.
        - Logging messages indicate where each output file is saved.

    """
    logging.info(f"Processing file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read()

    doc = nlp(text)
    sentence_ids, tokens, PoS, lemma, clean_tokens, clean_lemmas, depparse, head, constituency = [], [], [], [], [], [], [], [], []

    for sent_id, sentence in enumerate(doc.sentences):
        for word in sentence.words: 
            sentence_ids.append(sent_id)
            raw_token = word.text
            raw_lemma = word.lemma
            clean_token = raw_token.lower().translate(str.maketrans("","", string.punctuation)) #bisogna capire perché non ha eliminato tutta la punteggiatura, ""--> sono rimasti
            clean_lemma = raw_lemma.translate(str.maketrans("","",string.punctuation))
            tokens.append(raw_token)
            lemma.append(raw_lemma)
            clean_tokens.append(clean_token)
            clean_lemmas.append(clean_lemma)
            PoS.append(word.pos)
            depparse.append(word.deprel)
            head.append(sentence.words[word.head - 1].text if word.head > 0 else "ROOT")
            constituency.append(getattr(sentence, "constituency", None)) #Perché non funziona più?

    aoa_df = pd.read_excel("corpora\\ItAoA.xlsx", sheet_name="Database")
    aoa_df["M_AoA"] = aoa_df["M_AoA"].astype(str).str.replace(",", ".").astype(float)
    aoa_df["SD_AoA"] = aoa_df["SD_AoA"].astype(str).str.replace(",", ".").astype(float)    
    aoa_df["AoA_full"] = aoa_df.apply(lambda row: f"{row['M_AoA']:.2f} ± {row['SD_AoA']:.2f}", axis=1)
    aoa_dict = aoa_df.set_index("Ita_Word")["AoA_full"].to_dict()

    subtlex_df = pd.read_csv("corpora\\subtlex-it.csv", sep=";", encoding="cp1252")
    subtlex_df["zipf"] = subtlex_df["zipf"].astype(str).str.replace(",", ".").astype(float) #need number no string
    freq_dict = subtlex_df.set_index("wordform")["zipf"].to_dict()

    df = pd.DataFrame({
        "sentence_ids": sentence_ids,
        "tokens_no_punct": clean_tokens, #non sono utili entrambi, forse va bene anche solo questo no punct
        "lemma_no_punct": clean_lemmas,
        "PoS": PoS,
        "depparse": depparse,
        "head": head,
        "constituency": constituency
    })

    df["AoA(m+sd)"] = df["lemma_no_punct"].map(aoa_dict)
    df["Zipf_freq"] = df["tokens_no_punct"].map(freq_dict)


    name_base = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join (output_dir, name_base)
    os.makedirs(file_output_dir, exist_ok=True)
    csv_path = os.path.join(file_output_dir, f"{name_base}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")
    
    #eliminato per ora
    # # Frequency plot
    # lemma_freq = Counter(clean_lemmas)
    # sorted_lemma = lemma_freq.most_common()
    # labels, values = zip(*sorted_lemma)

    # plt.style.use("seaborn-v0_8")
    # plt.figure(figsize=(10, 6))
    # plt.bar(labels, values, color="powderblue")
    # plt.plot(labels, values, color='lightsteelblue', linestyle='-', linewidth=2)
    # step = max(1, len(labels) // 10)  
    # xticks_positions = list(range(0, len(labels), step))
    # xticks_labels = [labels[i] for i in xticks_positions]
    
    # plt.xticks(xticks_positions, xticks_labels, rotation=45)

    # plt.title(f"Lemma Frequency: {name_base}")
    # plt.xlabel("Lemma")
    # plt.ylabel("Frequency")
    # plt.tight_layout()

    # plot_path = os.path.join(file_output_dir, f"{name_base}_lemma_freq.png")
    # plt.savefig(plot_path)
    # plt.close()
    # logging.info(f"Saved plot: {plot_path}")

    #statistics
    num_tokens = len(clean_tokens)
    num_sentence = len(doc.sentences)
    num_lemmas = len(clean_lemma)
    num_types = len(set(clean_lemmas))
    ttr = num_types/num_lemmas if num_lemmas > 0 else 0
    sentence_lengths = [len(sentence.tokens) for sentence in doc.sentences]
    avg_sentence_length = sum(sentence_lengths)/len(sentence_lengths) if sentence_lengths else 0
    
    pos_count = Counter(PoS)
    function_pos = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PRON", "PART", "INTJ"}
    num_function_words = sum(pos_count[pos] for pos in function_pos)
    content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    num_content_words = sum(pos_count[pos] for pos in content_pos)
    num_verbs = pos_count.get("VERB", 0)
    num_adjectives = pos_count.get("ADJ", 0)
    num_nouns = pos_count.get("NOUN", 0) + pos_count.get("PROPN", 0)


    zipf_values = df["Zipf_freq"].dropna()
    zipf_mean = zipf_values.mean() #zipf 5 = parole comuni, zipf<3 --> parole rare
    zipf_std = zipf_values.std()
    rare_words = zipf_values[zipf_values <3]
    percent_rare = len(rare_words)/len(zipf_values) * 100
    #indice di leggibilità
    num_letters = sum(len(t) for t in clean_tokens if t)
    gulpease = 89 + (300 * num_sentence - 10 * num_letters)/ num_tokens if num_tokens > 0 else 0
    # https://it.wikipedia.org/wiki/Indice_Gulpease G <40 = testo difficile
    top_lemmas = Counter(clean_lemma).most_common(20)
    bigram_list = list(bigrams(clean_lemmas))
    top_bigrams = Counter(bigram_list).most_common(10)

    summary_stats = {
    "num_tokens": num_tokens,
    "num_sentences": num_sentence,
    "num_lemmas": num_lemmas,
    "num_types": num_types,
    "TTR": round(ttr, 3),
    "avg_sentence_length": round(avg_sentence_length, 2),
    "Zipf_mean": round(zipf_mean, 2),
    "Zipf_std": round(zipf_std, 2),
    "percent_rare_words": round(percent_rare, 2),
    "Gulpease_index": round(gulpease, 2),
    "top_lemmas": top_lemmas,
    "top_bigrams": [f"{a} {b}" for (a, b), _ in top_bigrams],
    "num_function_words": num_function_words,
    "num_content_words": num_content_words,
    "num_verbs": num_verbs,
    "num_adjectives": num_adjectives,
    "num_nouns": num_nouns

    }

   
    json_path = os.path.join(file_output_dir, f"{name_base}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved summary JSON: {json_path}")

    

