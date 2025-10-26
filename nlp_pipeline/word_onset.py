import os
import pandas as pd

df = pd.read_csv("data\\alignment_03\\02_03.csv", delimiter=";")

df.head()

df_valid = df[df["TOKEN"] != -1]

# Tieni solo la prima occorrenza di ciascun TOKEN
df_first = df_valid.drop_duplicates(subset=["TOKEN"], keep="first")

# Seleziona solo le colonne che ti interessano
result = df_first[["TOKEN", "Millisecondi", "correzioni", "ORT"]]

# Ordina per TOKEN (opzionale, se vuoi i numeri in ordine crescente)
result = result.sort_values(by="TOKEN").reset_index(drop=True)

print(result.head())

result.to_csv("output\\output_03\\prova_word_onset.csv")