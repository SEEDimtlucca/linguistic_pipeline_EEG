import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df_sur1 = pd.read_csv ("output\\provette\\01_1115\\Suprisal_01_1115_Gepù.csv")
df= pd.read_csv ("output\\provette\\01_1115\\Suprisal_01_1115.csv")

df_dis = pd.read_csv("output\\provette\\01_1115\\dissimilarity_01_1115.csv")


df_dis.columns
df_sur.columns
df_sur['surprisal'].median()
df_dis['semantic_dissimilarity'].median()
df_sur['surprisal'].describe()
df_dis['semantic_dissimilarity'].describe()

df_dis['sd_norm'] = (df_dis['semantic_dissimilarity'] - df_dis['semantic_dissimilarity'].min()) / \
                     (df_dis['semantic_dissimilarity'].max() - df_dis['semantic_dissimilarity'].min())

df_sur['su_norm'] = (df_sur['surprisal'] - df_sur['surprisal'].min()) / \
                     (df_sur['surprisal'].max() - df_sur['surprisal'].min())

df_sur['ent_norm'] = (df_sur['entropy'] - df_sur['entropy'].min()) / \
                     (df_sur['entropy'].max() - df_sur['entropy'].min())
var = 'surprisal'
sns.boxplot(df_sur[var])
plt.xlabel("Valore della variabile")
plt.ylabel("Densità")
plt.title("Distribuzione della variabile")
plt.show()

df_sur[df_sur['surprisal'] > 40 ]['word']
df_sur.describe()



plt.figure(figsize=(15,5))
plt.plot(df_sur['su_norm'], color='green', alpha=0.7, label="surp")
plt.plot(df_sur['ent_norm'], label='ent', color='yellow', alpha=0.7)
plt.xlabel("Word index")
plt.ylabel("Normalized value")
plt.title("Surprisal vs Entropy")
plt.legend()
plt.show()

import matplotlib.pyplot as plt



plt.figure(figsize=(8,6))
plt.scatter(df_sur['entropy'], df_sur['surprisal'], alpha=0.5)
plt.xlabel('Word-level Entropy (bits)')
plt.ylabel('Word Surprisal (bits)')
plt.title('Surprisal vs Word-level Entropy')
plt.show()


df['su_z'] = (df['surprisal'] - df['surprisal'].mean()) / df['surprisal'].std()
df['ent_z'] = (df['entropy'] - df['entropy'].mean()) / df['entropy'].std()
plt.figure(figsize=(12,5))
plt.hist(df['su_z'], bins=50, alpha=0.6, label='Surprisal')
plt.hist(df['ent_z'], bins=50, alpha=0.6, label='Word-level Entropy')
plt.xlabel('Value (bits)')
plt.ylabel('Count')
plt.legend()
plt.title('Distribution of Surprisal and Word-level Entropy')
plt.show()


plt.figure()
plt.plot(df['ent_z'], color="blue")
plt.plot(df['su_z'], color="purple")
plt.show()

#PROVETTE SINTASSI
from diaparser.parsers import Parser
parser = Parser.load("it_isdt.dbmdz-electra-xxl")
parser = Parser.load(lang="it")

with open("data\\texts_1115\\01_1115.txt", "r", encoding="utf-8") as infile:
    text = infile.read()

dataset = parser.predict(text, text= "it", prob=True)
dataset.sentences[1]
from diaparser.parsers import Parser

# Carica il modello pre‐addestrato per l’italiano

# Testo da analizzare
phrase = "Franci ha scritto un messaggio di esempio per testare il parser."

# Fai parsing diretto su testo grezzo (raw text)
dataset = parser.predict(phrase, text='it')

# Estrai la prima frase (che probabilmente è l’unica in questo caso)
sent = dataset.sentences[0]

# Stampa la struttura di dipendenza in formato CoNLL‐U
print(sent)

for token in sent.tokens:  
    print(token.form, token.lemma, token.upos, token.head, token.deprel)


