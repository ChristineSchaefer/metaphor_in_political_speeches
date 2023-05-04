import argparse


def parse():
    """
        Instantiate argument parser.
        Output:
            - parser object
    """
    parser = argparse.ArgumentParser(description="Metaphor Identification in Political Speeches")
    parser.add_argument("crawler_url", metavar="crawler_url", nargs="?", type=int,
                        help="Choose url for web crawler from list. Set to 0 if you don't want to use it.", choices=[0, 1, 2, 3])
    parser.add_argument("xml_path", metavar="xml_path", nargs="?", type=str,
                        help="Set path (as a string) to xml file with political speeches. "
                             "Set Empty string if you don't want to use it.")
    return parser
