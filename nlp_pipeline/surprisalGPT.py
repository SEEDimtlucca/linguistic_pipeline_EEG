import os
import torch
import math
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from nlp_pipeline.utils import reconstruct_words

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def calculate_surprisal (filepath: str, output_dir: str):
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
          aggregated at the word level thanks to the function 'reconstructed_words' from utils.py.
        - Output filenames are automatically derived from the input filename.
        - Logging messages indicate which file is being processed and where the
          resulting CSV is saved.
    """

    logging.info(f"Processing file: {filepath}")

    model_name = "LorenzoDeMattei/GePpeTto"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    model.eval()

    with open(filepath, "r", encoding="utf-8") as infile:
        text = infile.read().replace("\n", " ").replace("\r", " ")
   
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    values = [float("nan")] * len(tokens)

    for i in range(1, len(tokens)):
        token_id = int(input_ids[0, i].item())
        if (i - 1) >= log_probs.shape[1]:
            values[i] = float("nan")
            continue
        log_prob = float(log_probs[0, i - 1, token_id].item())
        surprisal = -log_prob / math.log(2)
        values[i] = surprisal

    words, word_surprisal = reconstruct_words(tokens, values, tokenizer, agg="sum")
    
    df = pd.DataFrame({"word": words, "surprisal": word_surprisal})
    
    name_base = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join (output_dir, name_base)
    os.makedirs(file_output_dir, exist_ok=True)
    csv_path = os.path.join(file_output_dir, f"Suprisal_{name_base}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")
