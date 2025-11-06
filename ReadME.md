# **Linguistic Features Extraction Pipeline for EEG Analysis**
This repository provides a Python-based pipeline for extracting linguistic features from texts, tailored for integrazion with EEG experiments. 


# **0 - years**
- 01_03 Il fatto è.
- 02_03 Il piccolo ragno tesse e tace.
- 03_03 Arrabbiato come un orso.
- 04_03 Lupetto mangia solo pastasciutta.
- 05_03 Cinque minuti di pace.
- 06_03 Il bruco molto affamato.
- 07_03 Il ciuccio di Nina.
- 08_03 Una casa per il mostro.
- 09_03 Chi me l'ha fatta in testa!
- 10_03 Orso buco.

# **3 - 7 years**
- 01_37 Il re parrucchiere. St01_B
- 02_37 La fata gattina. St02_B
- 03_37 I pirati smemorati. St03_B
- 04_37 L’ombrello asciutto. St04_B
- 05_37 Bianco e nero. St05_B
- 06_37 I tre affamati. St06_B
- 07_37 Le due befane. St07_B
- 08_37 Il principe Tonno e Abissina. St08_B
- 09_37 Una bambina senza nome.  St09_B
- 10_37 Le posate sposate.  St10_B

# **7-11 years**
- 01_711 La passeggiata di un distratto. St01_C
- 02_711 Il paese senza punta. St02_C
- 03_711 La guerra delle campane. St03_C
- 04_711 Una viola al polo Nord. St04_C
- 05_711 Giacomo di cristallo. St05_C
- 06_711 Promosso più due. St06_C
- 07_711 Il paese dei cani. St07_C
- 08_711 L'Apollonia della marmellata.  St08_C
- 09_711 Il muratore della Valtellina.  St09_C
- 10_711 Il re Mida.  St10_C

# **11-15 years**
- 01_1115 Il contadino astrologo. St01_D
- 02_1115 La camicia dell'uomo contento.St02_D
- 03_1115 Il cardellino. St04_D
- 04_1115 I tre linguaggi. St06_D
- 05_1115 Una goccia. St03_D
- 06_1115 Le precauzioni inutili contro le frodi. St05_D
- 07_1115 L'incantesimo della volpe. St07_D
- 08_1115 I tacchini non ringraziano.  St08_D
- 09_1115 Racconto per bambini cattivi.  St09_D
- 10_1115 Apocalisse. St10_D

## **Repository Structure**

- **\corpora**
    This folder contain two corpora used in the pipeline:
        - _ItAoA.xlsx_= Italian Age of Acquisition norms (*Montefinese M, Vinson D, Vigliocco G. and Ambrosini E. (2019) Italian Age of Acquisition Norms for a Large Set of Words (ItAoA). Front. Psychol. 10:278. doi: 10.3389/fpsyg.2019.00278*).
        It's possible to download at the following site: https://osf.io/3trg2/overview
        - _subtlex-it.csv_ = it's a frequency database for Italian words based on movie subtitles (free downdload: https://osf.io/zg7sc/overview)

- **\doc**

- **\data**
    This folder contain all the data used in this project.
        - *\phoneme_onset_03* 
        - *\phoneme_onset_37* 
        - *\phoneme_onset_711* 
        - *\phoneme_onset_1115*
        - *\texts_03* 
        - *\texts_37*
        - *\texts_711*
        - *\texts_1115*
        - *\word_onset_03* = generated 
        - *\word_onset_37*
        - *\word_onset_711*
        - *\word_onset_1115*

- **\nlp_pipeline**
    - *processor.py*
    - *semantic_dissimilarity.py*
    - *surprisalGPT.py*
    - *utils.py*
    - *word_onset.py*

- **\output**
    - *\output_03*:
        -*\01:03*
            - 01_03_summary.json
            - 01_03.csv
            - dissimilarity_01_03.csv
            - Surprisal_01_03.csv
    -
- **\venv** 

- .gitignore
- main.py
- ReadME.md (this)