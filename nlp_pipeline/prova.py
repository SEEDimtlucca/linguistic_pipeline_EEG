import pandas as pd
import matplotlib.pyplot as plt

df_sur = pd.read_csv ("output\\output_03\\01_03\\Suprisal_01_03.csv")

df_dis = pd.read_csv("output\\01_03\\dissimilarity_01_03.csv")

df_dis.columns
df_sur.columns

df_dis['sd_norm'] = (df_dis['semantic_dissimilarity'] - df_dis['semantic_dissimilarity'].min()) / \
                     (df_dis['semantic_dissimilarity'].max() - df_dis['semantic_dissimilarity'].min())

df_sur['su_norm'] = (df_sur['surprisal_bits'] - df_sur['surprisal_bits'].min()) / \
                     (df_sur['surprisal_bits'].max() - df_sur['surprisal_bits'].min())

# Grafico a linea lungo l'indice dei token (stupidissimo, senza allineamento preciso)
plt.figure(figsize=(15,5))
plt.plot(df_dis['sd_norm'], label='Semantic Dissimilarity', color='blue', alpha=0.7)
plt.plot(df_sur['su_norm'], label='Lexical Surprisal', color='red', alpha=0.7)
plt.xlabel("Word index (stupidissimo)")
plt.ylabel("Normalized value")
plt.title("Stupidissimo visual check: Surprisal vs Semantic Dissimilarity")
plt.legend()
plt.show()