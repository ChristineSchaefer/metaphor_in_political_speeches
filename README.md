# Metaphor Identification in Political Speeches

Required: [python 3.11](https://docs.python.org/3.11/contents.html)


## What is this repository for?
The repository is for the master thesis "Metaphern im Politischen Diskurs: Transformer-Modelle für
die Automatische Erkennung" where two transformer models - BERT with GCN and DistilBERT - were trained on 
a corpus of verbal multi-word-expressions to transfer the knowledge to detect verbal metaphors in political
speeches.


## Repo structure

```
|-- 📁 data
|   |-- 📁 annotations            #  json files with annotations and annotation guideline
|   |-- 📁 collections            #  MongoDB collections with all necessary documents
|   |-- 📁 crf                    #  saved crf model during training 
|   |-- 📁 logs                   #  every console output as txt-file
|   |-- 📁 plots                  #  training plots (confusion matrix, bar charts etc.)  
|-- 📁 src
|   |-- 📁 data_handler           #  contains models, controller etc. for data operations
|   |   |-- 📁 agreement
|   |   |-- 📁 controller
|   |   |-- 📁 models             #  database models
|   |   |-- 📁 utils
|   |   |-- main.py
|   |   |-- README.md
|   |-- 📁 mwe_metaphors          #  contains models, controller etc. for tranformer training, fine-tuning, evalutation etc.
|   |   |-- 📁 controller         #  main controller for BERT with GCN and DistilBERT
|   |   |-- 📁 models
|   |   |-- 📁 utils
|   |   |-- main.py
|   |   |-- README.md
|   |-- 📁 utils                  #  general utility methods 
|   |-- .env.example              #  example file for environment variables used in config.py
|   |-- config.py                 #  settings, setup etc. (initialization relevant constants)
|   |-- database.py
|-- .gitignore                      
|-- README.md
|-- requirements.txt
```

### Requirements
| data_handler                                               | mwe_metaphor                                                     | general                                                                           |
|------------------------------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [duden](https://pypi.org/project/duden/)                   | [evaluate](https://pypi.org/project/evaluate/)                   | [pydantic](https://docs.pydantic.dev/latest/)                                     |
| [requests](https://pypi.org/project/requests/)             | [numpy](https://numpy.org/)                                      | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| [beautifulsoup4](https://pypi.org/project/beautifulsoup4/) | [pandas](https://pandas.pydata.org/)                             | [scikit-learn](https://scikit-learn.org/stable/)                                  |
|                                                            | [pytorch](https://pytorch.org/)                                  | [pymongo](https://www.mongodb.com/docs/drivers/pymongo/)                          |
|                                                            | [transformers](https://huggingface.co/docs/transformers/index)   |                                                                                   |
|                                                            | [scipy](https://scipy.org/)                                      |                                                                                   |
|                                                            | [python-crfsuite](https://github.com/scrapinghub/python-crfsuite)   |                                                                                   |
|                                                            | [tqdm](https://pypi.org/project/tqdm/)                           |                                                                                   |
|                                                            | [spacy_conll](https://spacy.io/universe/project/spacy-conll)     |                                                                                   |
|                                                            | [spacy](https://spacy.io/)                                       |                                                                                   |
|                                                            | [matplotlib](https://matplotlib.org/)                            |                                                                                   |
|                                                            | [seaborn](https://seaborn.pydata.org/)                           |                                                                                   |
|                                                            | [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) |                                                                                   |

All requirements are listed in `requirements.txt`.


## Local setup
1) Install [python 3.11](https://docs.python.org/3.11/contents.html)
2) Clone repository with `git clone https://github.com/ChristineSchaefer/metaphor_in_political_speeches.git`
3) Create virtual environment with `python -m venv /path/to/new/virtual/environment`
4) Install requirements with `pip install -r requirements.txt`
5) Create `.env` file from `.env.example` at the same level and fill empty vars or change existing ones

### Database setup
1) Download and install [MongoDB Compass](https://www.mongodb.com/try/download/compass)
2) Connect to local database connection (URI e.g. `mongodb://localhost:27017`)
3) Create database and collections. Add data from `data/collections/` (each file is a collection)
4) Update database env vars (`DB_HOST`, `DB_NAME`, `DB_PORT`)

### Corpus setup
1) Clone repository with annotated VMWE with `git clone https://gitlab.com/parseme/sharedtask-data.git`
2) Update corpus env vars (`MWE_DIR`, `MWE_TRAIN`, `MWE_TEST`, `MWE_VAL`)

### Additional setup
1) Download spacy language model with `python -m spacy download de_core_news_sm`


## Available workflows
The repository offers two main applications:
1) Compilation of the database collections and expansion of political speeches with information on speakers, parties, etc. (`src/data_handler/`)
2) Fine-Tuning of transformer models for metaphor identification (`src/mwe_metaphor/`)

All files and steps are explained in the corresponding `README.md` in the folders.