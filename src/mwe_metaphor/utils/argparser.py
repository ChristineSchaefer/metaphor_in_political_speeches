import argparse


def parse():
    """
        Instantiate argument parser.
        Output:
            - parser object
    """
    parser = argparse.ArgumentParser(description="Metaphor Identification in Political Speeches - Classification")
    parser.add_argument("bert_gnc", metavar="bert_gnc", nargs="?", type=int,
                        help="Choose if you want to train and classify BERT with GNC. 0 = no, 1 = yes.",
                        choices=[0, 1])
    parser.add_argument("distilBert_finetuned", metavar="distilBert_finetuned", nargs="?",
                        type=int, help="Choose if you want to train and classify with fine-tuned DistilBERT. 0 = no, 1 = yes.",
                        choices=[0, 1])
    parser.add_argument("distilBert_baseline", metavar="distilBert_baseline", nargs="?",
                        type=int, help="Choose if you want to classify with DistilBERT. 0 = no, 1 = yes.",
                        choices=[0, 1])
    parser.add_argument("num_epochs", metavar="num_epochs", nargs="?", type=int,
                        help="Choose number of epochs to train DistilBERT.", choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("training", metavar="training", nargs="?", type=bool,
                        help="Choose if you want to train DistilBERT.", choices=[True, False])

    return parser
