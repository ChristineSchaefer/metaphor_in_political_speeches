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

### Models
- `Annotation`: has all relevant attributes for an annotated sentence, like analyzed verb with index, basic and contextual meaning and label
- `Politician`: information about potential used politician with name and party
- `Speech`: speech from DWDS corpus with corresponding speaker as Politician object
- `TroFi_Dataset`: transformed Annotation objects in TroFi structure

The corresponding MongoDB collections can be found in `data/collections/`.

### Controller
- CrawlerController: Offers three different web crawlers that can be used for the URLs listed in the main script.
- XMLReaderController: Reads a xml with potential political speeches from path and transform them to documents.
- CSVController: Reads a csv file of annotations, find the basic meaning by using the duden api and transform them to annotation documents.
- TrofiCollectionController: Transforms annotation documents into TroFi documents.
- AgreementController: Compute Cohen's and Fleiss' Kappa for evaluate the annotation process.

## Workflows
- Preprocessing
- Evaluation of annotation process