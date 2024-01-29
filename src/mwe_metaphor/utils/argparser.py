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
    parser.add_argument("training", metavar="training", nargs="?", type=int,
                        help="Choose if you want to train DistilBERT. 0 = no, 1 = yes.", choices=[0, 1])
    parser.add_argument("crf", metavar="crf", nargs="?",
                        type=int, help="Choose if you want to build a crf model and predict with it. 0 = no, 1 = yes.",
                        choices=[0, 1])

    return parser
