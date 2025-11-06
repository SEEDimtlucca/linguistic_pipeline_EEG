import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

df_sur1 = pd.read_csv ("output\\provette\\01_1115\\Suprisal_01_1115_Gepù.csv")
df_sur = pd.read_csv ("output\\provette\\01_1115\\Suprisal_01_1115.csv")

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

var = 'surprisal'
sns.boxplot(df_sur[var])
plt.xlabel("Valore della variabile")
plt.ylabel("Densità")
plt.title("Distribuzione della variabile")
plt.show()

df_sur[df_sur['surprisal'] > 40 ]['word']
df_sur.describe()



plt.figure(figsize=(15,5))
plt.plot(df_sur['surprisal'], color='green', alpha=0.7)
plt.plot(df_sur1['surprisal'], label='Lexical Surprisal_Old', color='yellow', alpha=0.7)
plt.xlabel("Word index")
plt.ylabel("Normalized value")
plt.title("Surprisal vs Semantic Dissimilarity")
plt.legend()
plt.show()