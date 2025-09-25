import os 
import logging
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import stanza


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

stanza.download("it")
nlp = stanza.Pipeline("it", processors= "tokenize, mwt, pos, lemma, depparse", use_gpu=False) 

def process_text_file (filepath, output_dir): 
    """
    Processes a single text file using the Stanza NLP pipeline and extracts linguistic features.

    Parameters:
        filepath (str): Path to the input .txt file.
        output_dir (str): Directory where output files (CSV and PNG) will be saved.

    Outputs:
        - A CSV file containing token-level linguistic features (lemma, PoS, dependency, etc.).
        - A PNG bar chart showing lemma frequency distribution.

    Notes:
        - The output filenames are automatically derived from the input filename.
        - Constituency parsing is included if available in the sentence object.
    """
    logging.info(f'Proccesing file:{filepath}')

    with open (filepath, "r", encoding="utf-8") as infile: 
        text = infile.read()
    
    doc = nlp(text)
    sentence_ids = []
    tokens = []
    PoS = []
    lemma = []
    depparse = []
    head = []
    constituency = []
    
    for sent_id, sentence in enumerate(doc.sentences):
        for word in sentence.words:
            sentence_ids.append(sent_id)
            tokens.append(word.text)
            lemma.append(word.lemma)
            PoS.append(word.pos)
            depparse.append(word.deprel)
            head.append(sentence.words[word.head - 1].text if word.head > 0 else "ROOT")
            constituency.append(sentence.constituency if hasattr(sentence, "constituency") else None)

    df = pd.DataFrame ({
    "sentence_ids": sentence_ids,
    "tokens": tokens,
    "lemma" : lemma,
    "PoS": PoS,
    "depparse": depparse,
    "head": head,
    "constituency": constituency})

    name_base = os.path.split (os.path.basename(filepath))[0]
    csv_path = os.path.join (output_dir, f"{name_base}_features.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")
    
    lemma_freq = Counter(lemma)
    sorted_lemma = lemma_freq.most_common()
    labels, values = zip(*sorted_lemma)
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color = "powderblue" )
    plt.plot(labels, values, color='lightsteelblue', linestyle='-', linewidth=2)
    xticks_positions = [0, len(labels)//2, len(labels)-1]
    xticks_labels = [labels[i] for i in xticks_positions]
    plt.xticks(xticks_positions, xticks_labels, rotation=45)
    plt.title("Lemma Frequency: {name_base}")
    plt.xlabel("Lemma")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{name_base}_lemma_freq.png")
    plt.savefig(plot_path)
    plt.close ()
    logging.info(f"Saved plot: {plot_path}")

