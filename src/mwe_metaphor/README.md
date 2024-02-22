# Fine-Tuning of Tranformer Models for Metaphor Identification

Required: [python 3.11](https://docs.python.org/3.11/contents.html)


## What is this folder for?
In this folder are the main steps to fine-tune transformer models that they learn 
how to detect metaphors in political speeches.

### Folder structure

```
|-- 📁 mwe_metaphors
|   |-- 📁 controller    
|   |   |-- bert_training_controller.py         # controller for DistilBERT baseline and Fine-Tuning
|   |   |-- crf_controller.py                   # controller for CRF baseline
|   |   |-- gcn_training_controller.py          # controller for BERT with GCN
|   |   |-- prediction_controller.py            # controller for testing DistilBERT baseline and with Fine-Tuning
|   |-- 📁 models   
|   |   |-- bert_with_gcn_model.py              # corresponding model to BERT with GCN
|   |   |-- dataset_model.py                    # helper model class for dataset objects that can be processed by Hugging Face Transformers
|   |   |-- gnc_model.py                        # corresponding model to BERT with GCN
|   |   |-- highway_model.py                    # corresponding model to BERT with GCN
|   |   |-- spacy_model.py                      # helper model for SpaCy language model
|   |-- 📁 utils  
|   |   |-- argparser.py                        # argument parser         
|   |   |-- text_utils.py                       # utility file for text processing     
|   |   |-- training_utils.py                   # corresponding utility training file for BERT with GCN
|   |   |-- tsvlib.py                           # helper file to process tsv sentences
|   |   |-- visualisation.py                    # helper file for visualisation of DistilBERT results
|   |-- main.py  
|   |-- README.md
```

### Controller
There are four controllers to control the different steps for this approach.
- `BertTrainingController`: Controller to manage the Fine-Tuning process of the DistilBERT model
- `PredictionController`: Controller to manage testing on DistilBERT model
- `BERTWithGCNTrainingController`: Controller to manage the Fine-Tuning and evaluation process of the BERT with GCN model
- `CRFController`: Controller for CRF baseline

The controllers use different models from `mwe_metaphor/models/` and `data_handler/models/` to fine-tune the models.
With arguments passed to the `main`-method in `main.py` the different workflows are started.

## Workflows
For activating the single flows you can pass arguments by running the main script.
The main script can be started with `python -m src.mwe_metaphor.main` and the available arguments can be 
shown with `-h` at the end of the command.

`python -m src.mwe_metaphor.main -h`

Use the following arguments to activate the different workflows:
- BERT with GCN: `1 0 0 0 0`
- CRF baseline: `0 0 0 0 1`
- DistilBERT baseline: `0 0 1 0 0`
- DistilBERT with Fine-Tuning: `0 1 0 1 0` 
  - you can use an already fine-tuned and locally saved model with: `0 1 0 0 0` (the latest saved model will be used)
  - when you choose Fine-Tuning with multiple epochs, you can set the number of epochs in the env vars
  - after Fine-Tuning the model and tokenizer will be saved locally in `/data/models/`

e.g. `python -m src.mwe_metaphor.main 0 1 0 1 0` for DistilBERT with Fine-Tuning.

If you want to change the classification modus from binary to multi-label you can set
the corresponding env var `MODUS` to `multi_label`.


### Required env vars for workflows
| VAR              | CRF  | BERT with GCN | DistilBERT |
|------------------|------| - | - |
| MWE_TRAIN        | x    | x | x |
| MWE_TEST         | x    |  | x |
| MWE_VAL          |      |  | x |
| MWE_DIR          | x    | x | x |
| METAPHOR_DIR     |      | x |  |
| DB_HOST          | x    | x | x |
| DB_PORT          | x    | x | x |
| DB_NAME          | x    | x | x |
| BATCH_TRAIN      |      | x | x |
| BATCH_TEST       |      | x |  |
| K                |      | x |  |
| EPOCHS           |      | x | x |
| NUM_TOTAL_STEPS  |      | x |  |
| NUM_WARMUP_STEPS |      | x | x |
| HEADS            |      | x |  |
| HEADS_MWE        |      | x |  |
| DROPOUT          |      | x |  |
| LANGUAGE_MODEL   |      | x | x |
| MODEL            |      |  | x |
| MODEL_DIR        |      |  | x |
| MODUS            |      |  | x |
| MAX_LEN          |      | x |  |


### Visualisation

When using the DistilBERT Fine-Tuning workflow with multiple training epochs a few visualizations are created in the process.
They will be saved in `data/plots/` and `data/logs/training_history/`.