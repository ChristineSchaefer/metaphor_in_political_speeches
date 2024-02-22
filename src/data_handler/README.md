# Data Handler for Corpora Creation

Required: [python 3.11](https://docs.python.org/3.11/contents.html)


## What is this folder for?
In this folder are the preprocessing steps to collect relevant information for a corpus
with annotated political speeches and to transform the data into usable MongoDB database 
documents. Also there are the steps to evaluate the annotation process.

### Folder structure

```
|-- 📁 data_handler
|   |-- 📁 agreement
|   |   |-- fleiss.py                           # utility file to compute fleiss' kappa
|   |-- 📁 controller    
|   |   |-- agreement_controller.py             # controller to evaluate annotation process      
|   |   |-- csv_reader.py                       # controller for reading csv file and create document from each row
|   |   |-- trofi_collection_controller.py      # controller to TroFi collection from annotations    
|   |   |-- web_crawler.py                      # different web crawler         
|   |   |-- xml_reader.py                       # xml reader to combine politician and speeches documents
|   |-- 📁 models    
|   |   |-- annotations.py                      # db model for annotations           
|   |   |-- politician.py                       # db model for politicians               
|   |   |-- speech.py                           # db model for speeches          
|   |   |-- trofi_dataset.py                    # db model for ToFi data
|   |-- 📁 utils  
|   |   |-- argparser.py                        # argument parser         
|   |-- main.py  
|   |-- README.md
```

### Controller