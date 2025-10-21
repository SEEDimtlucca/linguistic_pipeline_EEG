# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM


# def get_token_metrics (texts: list[str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, truncation=False) -> dict:
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     B = len(texts)
#     device = next(model.parameters()).device

#     inputs = tokenizer(texts, return_tensors = "pt", padding=True, truncation = truncation, return_length =True).to(device)
    
#     torch.cuda.empty_cache() #clear any leftover memory
    
#     with torch.no_grad(), torch.amp.autocast(device.type): #non ci interessa la loss
#         outputs = model(input_ids = inputs.input_ids, labels = inputs.input_ids)
#         logits = outputs.logits[:, :-1].float()

#         log_probs = torch.log_softmax(logits, dim=-1)
#         next_tokens = inputs.input_ids[:, 1:]
#         attention_mask = inputs.attention_mask[:, 1:]

#         token_log_probs = log_probs.gather (-1, next_tokens.unsqueeze(-1)).squeeze(-1)

#         surprisals = -token_log_probs * attention_mask
#         null_first = torch.full((B, 1), float('nan'), device=device)
#         surprisals = torch.cat((null_first, surprisals), 1)
                
#         tokens = [
#             tokenizer.convert_ids_to_tokens(seq[:mask.sum()].tolist())
#             for seq, mask in zip(inputs.input_ids, inputs.attention_mask)]


#         assert all(len(s) == n for s, n in zip(tokens, inputs.length))
#         return {
#             'tok_str':  tokens,  # list[list[str]] jagged shape [B, Tb<=T]
#             'tok_surp': surprisals.cpu(),  # tensor[B, T]
        
#             'tok_attn': attention_mask.cpu(),  # tensor[B, T]
    
#             'seq_len':  inputs.length.cpu(),  # tensor[B]
#             'vocab_size': len(tokenizer),  # int
#         }

# with open(r"data\texts_03\01_03.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# model = AutoModelForCausalLM.from_pretrained("LorenzoDeMattei/GePpeTto").to(device)
# tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto", clean_up_tokenization_spaces = True)

# results = get_token_metrics([text], model, tokenizer)
# tokens = results["tok_str"][0]
# surprisals = results["tok_surp"][0].tolist()

# words = []
# word_surps = []
# curr_word = []
# curr_surps = []

# for tok, s in zip(tokens, surprisals):
#     if torch.isnan(torch.tensor(s)):
#         continue

#     if tok.startswith("Ġ"):
#         if curr_word:
#             words.append(tokenizer.convert_tokens_to_string(curr_word).strip())
#             word_surps.append(sum(curr_surps))
            
#         curr_word = [tok]
#         curr_surps= [s]
#     else:
#         curr_word.append(tok)
#         curr_surps.append(s)

# if curr_word:
#     words.append(tokenizer.convert_tokens_to_string(curr_word).strip())
#     word_surps.append(sum(curr_surps))

# df = pd.DataFrame({
#     "word": words,
#     "surprisal": word_surps})

# df.to_csv("surprisal_01_03_pt3.csv", index=False)

# from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
# import torch
# import math
# import pandas as pd
# import re 

# # === 1. Carichiamo il modello GPT-2 italiano (GePpeTto) ===
# model_name = "LorenzoDeMattei/GePpeTto"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# model.eval()

# # === 2. Frase di esempio ===
# with open ("data\\texts_03\\01_03.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# # === 3. Tokenizzazione e input ===
# inputs = tokenizer(text, return_tensors="pt")
# input_ids = inputs["input_ids"]

# # === 4. Otteniamo le logits e calcoliamo log-probabilità ===
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits
# log_probs = torch.log_softmax(logits, dim=-1)

# # === 5. Calcolo surprisal a livello di parola ===
# tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# # Ricreiamo la frase token per token, raggruppando in base agli spazi
# surprisal_per_word = []
# current_word = ""
# current_surprisal = 0.0

# punct_pattern = re.compile(r"^[\.\,\!\?\;\:\\\"\'\)\(\[\]\-\–\—]+$")


# for i in range(1, len(tokens)):  # skip del primo token (inizio)
#     token = tokens[i]
#     token_id = input_ids[0, i]

#     # surprisal subtoken (in bit)
#     log_prob = log_probs[0, i - 1, token_id].item()
#     surprisal = -log_prob / math.log(2)
    
#     decoded_token = tokenizer.convert_tokens_to_string([token])
 
#     if decoded_token.startswith("Ġ") or token.startswith("▁"):
#         if current_word:
#             # aggiungi parola precedente se non è solo punteggiatura
#             cleaned_word = current_word.lstrip("Ġ▁")
#             if cleaned_word and not punct_pattern.match(cleaned_word):
#                 surprisal_per_word.append((cleaned_word, current_surprisal))
#         # inizializza nuova parola
#         current_word = token.lstrip("Ġ▁")
#         current_surprisal = surprisal
#     else:
#         # concatena subtoken allo stesso word
#         current_word += token
#         current_surprisal += surprisal

# # # aggiungi ultima parola
# # if current_word:
# #     cleaned_word = current_word.lstrip("Ġ▁")
# #     if cleaned_word and not punct_pattern.match(cleaned_word):
# #         surprisal_per_word.append((cleaned_word, current_surprisal))

# # === 6. Stampa risultati ===
# print(surprisal_per_word)
# df = pd.DataFrame(surprisal_per_word, columns=["word", "surprisal_bits"])
# df.to_csv("surprisal_outputMinerva.csv", index=False, encoding="utf-8-sig")


# reconstructed_text = tokenizer.convert_tokens_to_string(tokens)
# print(reconstructed_text)
# reconstructed_text.type
# print (tokens)

#codice che calcola bene il surprisal MA con apostrofo
import torch, math, pandas as pd, re
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 1. Modello e tokenizer ===
model_name = "LorenzoDeMattei/GePpeTto"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# === 2. Testo ===
with open("data/texts_03/02_03.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("\n", " ").replace("\r", " ")

# === 3. Tokenizzazione ===
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# === 4. Calcolo log-prob ===
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
log_probs = torch.log_softmax(logits, dim=-1)

# === 5. Ottieni token e calcola surprisal ===
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

surprisal_per_word = []
current_tokens = []
current_surprisal = 0.0
punct_pattern = re.compile(r"^[\.\,\!\?\;\:\\\"\'\)\(\[\]\-\–\—]+$")

for i in range(1, len(tokens)):
    token = tokens[i]
    token_id = input_ids[0, i]
    log_prob = log_probs[0, i - 1, token_id].item()
    surprisal = -log_prob / math.log(2)

    # chiudi la parola se il token inizia con spazio o è solo punteggiatura
    if token.startswith("Ġ") or punct_pattern.match(token):
        if current_tokens:
            # converti subtoken in parola
            word = tokenizer.convert_tokens_to_string(current_tokens).strip()
            word = word.strip('"')  # rimuove eventuali virgolette
            if word:
                surprisal_per_word.append((word, current_surprisal))
        current_tokens = [token]  # nuova parola
        current_surprisal = surprisal
    else:
        # subtoken: continua la parola
        current_tokens.append(token)
        current_surprisal += surprisal

# aggiungi ultima parola
if current_tokens:
    word = tokenizer.convert_tokens_to_string(current_tokens).strip()
    word = word.strip('"')
    if word:
        surprisal_per_word.append((word, current_surprisal))

# === 6. Salva ===
df = pd.DataFrame(surprisal_per_word, columns=["word", "surprisal_bits"])
df.to_csv("provetta.csv", index=False, encoding="utf-8-sig")

print(df.head(20))
# Filtra solo le parole che NON sono punteggiatura
filtered_surprisal = [(word, s) for word, s in surprisal_per_word if not punct_pattern.fullmatch(word)]

# Salva nel CSV
df = pd.DataFrame(filtered_surprisal, columns=["word", "surprisal_bits"])
df.to_csv("surprisal_output_GePpeTto_clean.csv", index=False, encoding="utf-8-sig")


#ULTIMA PROVA --> codice che separa bene dopo l'apostrofo, MA surprisal basso nella parola dopo l'apostrofo
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
import pandas as pd
import re

# === 1. Carichiamo il modello GPT-2 italiano (GePpeTto) ===
model_name = "LorenzoDeMattei/GePpeTto"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# === 2. Frase di esempio ===
with open("data\\texts_03\\01_03.txt", "r", encoding="utf-8") as f:
    text = f.read()

# sostituisci a capo con spazio per non rompere i token
text = text.replace("\n", " ").replace("\r", " ")

# === 3. Tokenizzazione e input ===
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# === 4. Otteniamo le logits e calcoliamo log-probabilità ===
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
log_probs = torch.log_softmax(logits, dim=-1)

# === 5. Calcolo surprisal a livello di parola ===
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
surprisal_per_word = []
current_tokens = []
current_surprisal = 0.0
#punct_pattern = re.compile(r"^[\.\,\!\?\;\:\\\"\'\)\(\[\]\-\–\—]+$")
punct_pattern = re.compile(r"\^\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\{\|\}\~")


for i in range(1, len(tokens)):
    token = tokens[i]
    token_id = input_ids[0, i]
    log_prob = log_probs[0, i - 1, token_id].item()
    surprisal = -log_prob / math.log(2)

    # gestisce token apostrofo BPE
    if token == "âĢĻ":
    # chiudi parola precedente aggiungendo l'apostrofo
        if current_tokens:
            current_tokens[-1] += "'"  # l' , d' , senz' etc.
            surprisal_per_word.append((tokenizer.convert_tokens_to_string(current_tokens).strip(), current_surprisal))
    # reset per parola successiva
        current_tokens = []
        current_surprisal = 0.0
        continue


    # chiudi parola se:
    # - token inizia con spazio
    # - token è punteggiatura
    if token.startswith("Ġ") or punct_pattern.match(token):
        if current_tokens:
            word = tokenizer.convert_tokens_to_string(current_tokens).strip().strip('"')
            if word:
                surprisal_per_word.append((word, current_surprisal))
        # inizia nuova parola
        current_tokens = [token]
        current_surprisal = surprisal
    else:
        # subtoken della stessa parola
        current_tokens.append(token)
        current_surprisal += surprisal

# aggiungi ultima parola
if current_tokens:
    word = tokenizer.convert_tokens_to_string(current_tokens).strip().strip('"')
    if word:
        surprisal_per_word.append((word, current_surprisal))

# === 6. Rimuove solo le parole che sono punteggiatura dal CSV finale ===
filtered_surprisal = [(word, s) for word, s in surprisal_per_word if not punct_pattern.fullmatch(word)]

# === 7. Salva CSV finale ===
df = pd.DataFrame(filtered_surprisal, columns=["word", "surprisal_bits"])
df.to_csv("surprisal_output_GePpeTto_provaRE.csv", index=False, encoding="utf-8-sig")


# === 8. Stampa per controllo ===
for w, s in filtered_surprisal[:30]:  # stampo solo le prime 30 parole
    print(f"{w}: {s:.3f}")

import string

print (string.punctuation)