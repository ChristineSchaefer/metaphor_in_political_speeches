# metaphor_in_political_speeches

Required: [python 3.11](https://docs.python.org/3.11/contents.html)

## What is this repository for?
The repository is for the master thesis "Metaphern im Politischen Diskurs: Transformer-Modelle für
die Automatische Erkennung" where two transformer models - BERT with GCN and DistilBERT - were trained on 
a corpus of verbal multi-word-expressions to transfer the knowledge to detect verbal metaphors in political
speeches.

## Repo structure

```
|-- 📁 data
|   |-- 📁 annotations            #  json files with annotations
|   |-- 📁 collections            #  MongoDB collections with all necessary documents
|   |-- 📁 logs                   #  every console output as txt-file
|   |-- 📁 plots                  #  training plots (confusion matrix, bar charts etc.)  
|-- 📁 docs
|-- 📁 src
|   |-- __init__.py
|   |-- 📁 controllers             # contains business logic in python packages usable as singletons (MVC)
|   |-- 📁 models                  # dataclasses, schemas for our DBs
|   |-- 📁 routes                  # enpoints callable via http or ws namespaces (Views in the MVC paradigma)      
|   |-- 📁 serializers             # models parsing and validating views' input/output  
|   |-- config.py                  # settings (dotenv), logging setup etc. (initialization relevant constants)
|   |-- constants.py               # run time relevant constants, importable from everywhere 
|   |-- exceptions.py              # custom exceptions that can be raised anywhere in the project 
|   |-- main.py                    # contains the fastAPI app and ODM initialization
|   `-- utils.py                   # top level functions that can be used from everywhere but constants.py
|-- 📁 deployment
|   |-- .env                       # git-ignored
|   |-- .env.example               # create your .env based on this example
|   |-- Dockerfile
|   |-- cloudbuild-dev.yaml
|   `-- cloudbuild-prod.yaml
|-- 📁 templates
|   |-- __init__.py
|   |-- wsdocs.html.jinja
|-- 📁 docs                        # Documentation (diagrams, tutorials, glossary...)
|   `-- diagrams                   # UML diagrams for this project, using puml language   
|-- 📁 logs                        # git-ignored
|-- 📁 tests
|   |-- __init__.py
|   |-- conftest.py            # configuration of fixtures and pytest 
|   `-- test_<...>.py          # unit-, models-, routes- tests
|-- .gitignore                      
|-- asyncapi.yaml                  # API Specification of the event-driven part (websockets)                     
|-- client.py                      # A socketio client, useful for manual testing of events                    
|-- pyproject.toml                 # Project wide setup (e.g. linter options)                      
|-- pytest.ini                     # test setup for the whole project
|-- README.md
|-- CHANGELOG.md
|-- semver.py                      # script to update version for deployment
`-- requirements.txt
```