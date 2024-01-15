import argparse


def parse():
    """
        Instantiate argument parser.
        Output:
            - parser object
    """
    parser = argparse.ArgumentParser(description="Metaphor Identification in Political Speeches - Data Management")
    parser.add_argument("crawler_url", metavar="crawler_url", nargs="?", type=int,
                        help="Choose url for web crawler from list. Set to 0 if you don't want to use it.",
                        choices=[0, 1, 2, 3])
    parser.add_argument("xml_path", metavar="xml_path", nargs="?", type=str,
                        help="Set path (as a string) to folder with xml files of political speeches. "
                             "Set empty string if you don't want to use it.")
    parser.add_argument("csv_path", metavar="csv_path", nargs="?", type=str,
                        help="Set path (as a string) to folder with csv file of annotations. "
                             "Set empty string if you don't want to use it.")
    parser.add_argument("db_job", metavar="db_job", nargs="?", type=int,
                        help="Choose if you want to work on database jobs. 0 = no, 1 = yes.", choices=[0, 1])
    parser.add_argument("agreement", metavar="agreement", nargs="?", type=int,
                        help="Choose if you want to compute the inner-annotator-agreement. 0 = no, 1 = yes.",
                        choices=[0, 1])
    return parser