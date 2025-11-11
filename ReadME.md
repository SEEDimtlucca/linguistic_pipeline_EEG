# **Linguistic Features Extraction Pipeline for EEG Analysis**
This repository provides a Python-based pipeline for extracting linguistic features from texts, tailored for integraTion with EEG experiments. 

## **Repository Structure**

- **\corpora**
Contains two corpora used in the pipeline:
    - `ItAoA.xlsx`= Italian Age of Acquisition norms (*Montefinese M, Vinson D, Vigliocco G. and Ambrosini E. (2019) Italian Age of Acquisition Norms for a Large Set of Words (ItAoA). Front. Psychol. 10:278. doi: 10.3389/fpsyg.2019.00278*).
      Download: https://osf.io/3trg2/overview
    - `subtlex-it.csv` = Frequency database for Italian words based on movie subtitles.
      Downdload: https://osf.io/zg7sc/overview

- **\doc**
Contains documentation files:
    - `all_summary.pdf` = Explanation of all linguistic statistics extracted from the texts.
    - `all_summary.xlsx` = Tabular file containing all linguistic statistics.
    - TO ADD: .pdf file for all the features extracted 

- **\data**
Contains all the experimental data:
    - *\phoneme_onset** = .xlsx files with phoneme onsets (in samples) for each story.
    - *\texts** = texts files of the stories in .txt formats.
    - *\word_onset** = .csv files with word onsets (in samples) for each story.

- **\nlp_pipeline**
Python modules that implement the features extraction:
    - `processor.py`
    - `semantic_dissimilarity.py`
    - `surprisalGPT.py`
    - `utils.py`

- **\output**
All the subfolders in 'output' folder are structured in the following way: one subfolder for each group of age stories and one subfolder for each story of the group that cointains the following files:
    - `<story_id>_summary.json`
    - `<story_id>.csv`
    - `dissimilarity_<story_id>.csv`
    - `Surprisal_<story_id>.csv`

- **Other files**
    - `main.py` = Main script to run the pipeline
    - `ReadME.md`= This file


All the stories, stored into the 'data' forlder, are diveded into four groups based on the age.
Titles and correponding codes are listed below.

### **0 - 3 years**
- 01_03 Il fatto è. St01_A
- 02_03 Il piccolo ragno tesse e tace. St02_A
- 03_03 Arrabbiato come un orso. St03_A
- 04_03 Lupetto mangia solo pastasciutta. St04_A
- 05_03 Cinque minuti di pace. St05_A
- 06_03 Il bruco molto affamato. St06_A
- 07_03 Il ciuccio di Nina. St07_A
- 08_03 Una casa per il mostro.  St08_A
- 09_03 Chi me l'ha fatta in testa!  St09_A
- 10_03 Orso buco. St10_A

### **3 - 7 years**
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

### **7-11 years**
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

### **11-15 years**
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
