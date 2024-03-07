# Metaphor Identification in Political Speeches

Required: [python 3.11](https://docs.python.org/3.11/contents.html)

Implemented on macOS.


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
|   |-- 📁 tests                  #  files for test purposes
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
|                                                            | [accelerate](https://huggingface.co/docs/accelerate/index) |                                                                                   |

All requirements are listed in `requirements.txt`.


## Local setup
1) Install [python 3.11](https://docs.python.org/3.11/contents.html)
2) Clone repository with `git clone https://github.com/ChristineSchaefer/metaphor_in_political_speeches.git`
   - if error occurs you can try `git clone --depth=1 https://github.com/ChristineSchaefer/metaphor_in_political_speeches.git`
   or download the repository as zip 
3) Open repo and create virtual environment with `python -m venv /path/to/new/virtual/environment` and activate it
4) Install requirements with `pip install -r requirements.txt`
5) Create `.env` file from `.env.example` at the same level (`src`) and fill empty vars or change existing ones
   - there are some default values in `.env.example`, you can use these if they match or updated with new values

### Corpus setup
1) Clone repository with annotated VMWE with `git clone https://gitlab.com/parseme/sharedtask-data.git`
2) Update following corpus env vars:
   - `MWE_DIR`: path to folder, e.g. /Users/testuser/sharedtask-data/1.2/DE
   - `MWE_TRAIN`, `MWE_TEST` and `MWE_VAL`: name of the file

### Database setup
1) Running MongoDB community instance (see: [installation guide](https://www.mongodb.com/docs/manual/administration/install-community/))
   - make sure you have a MongoDB service running locally on port 27017 (e.g. activate with macos: 'brew services mongodb-community', ubuntu: 'sudo systemctl start mongod')
2) Download and install [MongoDB Compass](https://www.mongodb.com/try/download/compass)
3) Connect to local database connection (URI e.g. `mongodb://localhost:27017`)
4) Create database and collections. Either
   - add data from `data/collections/` (each file is a collection) or
   - set env var `INIT_DB` to `True` (if you choose this option, **you should only execute this option once**, 
   i.e. set the var to `False` again after the first time, otherwise the data will be written to the database twice)
5) Update database env vars (`DB_HOST`, `DB_NAME`, `DB_PORT`)

### Additional setup
1) Download spacy language model with `python -m spacy download de_core_news_sm`


## Available workflows
The repository offers two main applications:
1) Compilation of the database collections and expansion of political speeches with information on speakers, parties, etc. (`src/data_handler/`)
2) Fine-Tuning of transformer models for metaphor identification (`src/mwe_metaphor/`)

All files and steps are explained in the corresponding `README.md` in the folders.