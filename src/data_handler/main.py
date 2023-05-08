from src.data_handler.controller.xml_reader import XMLReaderController
from src.utils import argparser
from src.data_handler.controller.web_crawler import CrawlerController

urls = ["https://goslar-gegen-rechtsextremismus.de/html/afd-sprueche.php",
        "https://www.bundestag.de/parlament/geschichte/bundestagspraesidenten_seit_1949/bundestagspraesidenten_seit_1949-196684",
        "https://de.wikipedia.org/wiki/Liste_der_deutschen_Regierungsmitglieder_seit_1949"]


def main_data_handling(arguments):
    if arguments.crawler_url != 0:
        cc = CrawlerController(url=urls[arguments.crawler_url-1], index=arguments.crawler_url)
        cc.get_card_content()

    elif arguments.xml_path:
        # TODO keine einzelne Datei, sondern Ordner mit Dateien, die alle durchgelaufen werden sollen
        xrc = XMLReaderController(path=arguments.xml_path)
        xrc.parse()

    else:
        print("Please set valid arguments.")


if __name__ == "__main__":
    parser = argparser.parse()
    main_data_handling(parser.parse_args())

