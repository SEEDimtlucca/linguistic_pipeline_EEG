import os
import torch
import math
import pandas as pd
import re
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def calculate_surprisal (filepath, output_dir):
    """
    Calculates token-level surprisal values for a text file using the GePpeTto
    Italian language model (a GPT-based causal language model).

    For each input file, a dedicated subfolder is created (named after the file, e.g. '01_03')
    where the output CSV is saved.

    Parameters:
        filepath (str): Path to the input .txt file.
        output_dir (str): Root directory where output subfolders will be created.

    Outputs:
        - A CSV file containing word-level surprisal values in bits:
          columns include the word and its surprisal score.
          
    Notes:
        - The function uses the Hugging Face `AutoTokenizer` and `AutoModelForCausalLM`
          to tokenize the text and compute log-probabilities.
        - Surprisal is computed as the negative log2 probability of each token,
          aggregated at the word level.
        - Punctuation tokens are filtered out from the final output.
        - Output filenames are automatically derived from the input filename.
        - Logging messages indicate which file is being processed and where the
          resulting CSV is saved.
    """

    logging.info(f"Processing file: {filepath}")

    model_name = "LorenzoDeMattei/GePpeTto"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read()

    text = text.replace("\n", " ").replace("\r", " ")
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    surprisal_per_word = []
    current_tokens = []
    current_surprisal = 0.0
    punct_pattern = re.compile(r"\^\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\{\|\}\~")

    for i in range(1, len(tokens)):
        token = tokens[i]
        token_id = input_ids[0, i]
        log_prob = log_probs[0, i - 1, token_id].item()
        surprisal = -log_prob / math.log(2)

        if token == "âĢĻ":
            if current_tokens:
                current_tokens[-1] += "'" 
                surprisal_per_word.append((tokenizer.convert_tokens_to_string(current_tokens).strip(), current_surprisal))
            current_tokens = []
            current_surprisal = 0.0
            continue

        if token.startswith("Ġ") or punct_pattern.match(token):
            if current_tokens:
                word = tokenizer.convert_tokens_to_string(current_tokens).strip().strip('"')
                if word:
                    surprisal_per_word.append((word, current_surprisal))
            current_tokens = [token]
            current_surprisal = surprisal
        else:
            current_tokens.append(token)
            current_surprisal += surprisal

    if current_tokens:
        word = tokenizer.convert_tokens_to_string(current_tokens).strip().strip('"')
        if word:
            surprisal_per_word.append((word, current_surprisal))

    filtered_surprisal = [(word, s) for word, s in surprisal_per_word if not punct_pattern.fullmatch(word)]

    df = pd.DataFrame(filtered_surprisal, columns=["word", "surprisal_bits"])
    
    name_base = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join (output_dir, name_base)
    os.makedirs(file_output_dir, exist_ok=True)
    csv_path = os.path.join(file_output_dir, f"Suprisal_{name_base}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")