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
|   |   |-- trofi_collection_controller.py      # controller to create TroFi collection from annotations    
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
- `CrawlerController`: Offers two different web crawlers that can be used for the URLs listed in the main script.
- `XMLReaderController`: Reads a xml with potential political speeches from path and transform them to documents.
- `CSVController`: Reads a csv file of annotations, find the basic meaning by using the duden api and transform them to Annotation documents.
- `TrofiCollectionController`: Transforms Annotation documents into TroFi documents.
- `AgreementController`: Compute Cohen's and Fleiss' Kappa for evaluate the annotation process.

## Workflows
- Preprocessing: For the preprocessing process for enriching the data with additional information like speaker with party
etc. there are several steps to collect the information from external websites.
The Preprocessing step uses the `CrawlerController`, `XMLReaderController`, `CSVController` and `TrofiCollectionController`
to do the following things:
  - Collect information from web about current active german politicians and their parties and save in db
  - Save speeches in db and set speaker from Politician objects
  - Update annotations with basic meaning, index etc.
  - Transform annotations to TroFi data
- Evaluation of annotation process: The annotations in `data/annotations/` will be compared to compute the IAA with
Cohen's and Fleiss' Kappa with `AgreementController`.
  - Cohen's Kappa will be computed for each annotator combination.

For activating the single flows you can pass arguments by running the main script.
The main script can be started with `python -m src.data_handler.main` and the available arguments can be 
shown with `-h` at the end of the command.

`python -m src.data_handler.main -h`

When using the data from `data/collections/` you don't need to run the four preprocessing steps.
The collection data contains the results from the controller, you can only import the data to your 
MongoDB database.

Use the following example arguments to activate the different workflows:
- Web Crawler: `1 "" "" 0 0`
  - there are two different crawler for URLs listed in `main.py`
  - used for collect information about politician and fill db
- XML Reader: `0 "path/to/xml/" "" 0 0`
  - used for transform the corpus of political speeches from https://politische-reden.eu/ into documents 
  (after download the xml files to a local folder)
  - you can test this flow with the folder `data/tests/`
- CSV Reader: `0 "" "path/to/csv/file.csv" 0 0`
  - you can test this flow with the file `data/tests/annotations.csv`
- TroFi Transformation: `0 "" "" 1 0`
- Annotation Evaluation: `0 "" "" 0 1`

e.g. `python -m src.data_handler.main 0 "" "" 0 1` for evaluation of the annotation process.

### Required env vars for workflows
| VAR              | Web Crawler | XML Reader | CSV Reader | TroFi Tranformer | Agreement |
|------------------|-------------|-----|-----|-----|-----|
| DB_HOST          | x           | x | x | x | x |
| DB_PORT          | x           | x | x | x | x |
| DB_NAME          | x           | x | x | x | x |