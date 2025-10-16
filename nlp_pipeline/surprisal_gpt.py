import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_token_metrics (texts: list[str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, truncation=False) -> dict:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    B = len(texts)
    device = next(model.parameters()).device

    inputs = tokenizer(texts, return_tensors = "pt", padding=True, truncation = truncation, return_length =True).to(device)
    
    torch.cuda.empty_cache() #clear any leftover memory
    
    with torch.no_grad(), torch.amp.autocast(device.type): #non ci interessa la loss
        outputs = model(input_ids = inputs.input_ids, labels = inputs.input_ids)
        logits = outputs.logits[:, :-1].float()

        log_probs = torch.log_softmax(logits, dim=-1)
        next_tokens = inputs.input_ids[:, 1:]
        attention_mask = inputs.attention_mask[:, 1:]

        token_log_probs = log_probs.gather (-1, next_tokens.unsqueeze(-1)).squeeze(-1)

        surprisals = -token_log_probs * attention_mask
        null_first = torch.full((B, 1), float('nan'), device=device)
        surprisals = torch.cat((null_first, surprisals), 1)
                
        tokens = [
            tokenizer.convert_ids_to_tokens(seq[:mask.sum()].tolist())
            for seq, mask in zip(inputs.input_ids, inputs.attention_mask)]


        assert all(len(s) == n for s, n in zip(tokens, inputs.length))
        return {
            'tok_str':  tokens,  # list[list[str]] jagged shape [B, Tb<=T]
            'tok_surp': surprisals.cpu(),  # tensor[B, T]
        
            'tok_attn': attention_mask.cpu(),  # tensor[B, T]
    
            'seq_len':  inputs.length.cpu(),  # tensor[B]
            'vocab_size': len(tokenizer),  # int
        }

with open(r"data\texts_03\01_03.txt", "r", encoding="utf-8") as f:
    text = f.read()

device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained("LorenzoDeMattei/GePpeTto").to(device)
tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto", clean_up_tokenization_spaces = True)

results = get_token_metrics([text], model, tokenizer)
tokens = results["tok_str"][0]
surprisals = results["tok_surp"][0].tolist()

words = []
word_surps = []
curr_word = []
curr_surps = []

for tok, s in zip(tokens, surprisals):
    if torch.isnan(torch.tensor(s)):
        continue

    if tok.startswith("Ġ"):
        if curr_word:
            words.append(tokenizer.convert_tokens_to_string(curr_word).strip())
            word_surps.append(sum(curr_surps))
            
        curr_word = [tok]
        curr_surps= [s]
    else:
        curr_word.append(tok)
        curr_surps.append(s)

if curr_word:
    words.append(tokenizer.convert_tokens_to_string(curr_word).strip())
    word_surps.append(sum(curr_surps))

df = pd.DataFrame({
    "word": words,
    "surprisal": word_surps})

df.to_csv("surprisal_01_03_pt3.csv", index=False)
