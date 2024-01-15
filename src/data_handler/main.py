from src.data_handler.controller.agreement_controller import AgreementController
from src.data_handler.controller.trofi_collection_controller import TrofiCollectionController
from src.data_handler.controller.csv_reader import CSVController
from src.data_handler.controller.xml_reader import XMLReaderController
from src.data_handler.models.annotations import Annotation
from src.data_handler.utils import argparser
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

    elif arguments.csv_path:
        cc = CSVController(path=arguments.csv_path)
        cc.create_annotation_object()

    elif arguments.db_job == 1:
        annotations = Annotation.find()
        cc = TrofiCollectionController(annotations=annotations)
        cc.create_trofi_object_and_save_in_collection()

    elif arguments.agreement == 1:
        ac = AgreementController()
        ac.compute_fleiss_agreement()
        ac.compute_cohen_agreement()

        agreement1, disagreement1 = ac.get_cohen_examples(0, 1)
        print(f"annotator 1 and 2 agreement for metaphor: {len(agreement1['is_metaphor'])} \n {[a.id for a in agreement1['is_metaphor']]}")
        print(
            f"annotator 1 and 2 agreement for no metaphor: {len(agreement1['no_metaphor'])} \n {[a.id for a in agreement1['no_metaphor']]}")
        print(
            f"annotator 1 and 2 disagreement: {len(disagreement1)} \n {[d.id for d in disagreement1]}")

        agreement2, disagreement2 = ac.get_cohen_examples(0, 2)
        print(f"annotator 1 and 3 agreement for metaphor: {len(agreement2['is_metaphor'])} \n {[a.id for a in agreement2['is_metaphor']]}")
        print(
            f"annotator 1 and 3 agreement for no metaphor: {len(agreement2['no_metaphor'])} \n {[a.id for a in agreement2['no_metaphor']]}")
        print(
            f"annotator 1 and 3 disagreement: {len(disagreement2)} \n {[d.id for d in disagreement2]}")

        agreement3, disagreement3 = ac.get_cohen_examples(2, 1)
        print(f"annotator 3 and 2 agreement for metaphor: {len(agreement3['is_metaphor'])} \n {[a.id for a in agreement3['is_metaphor']]}")
        print(
            f"annotator 3 and 2 agreement for no metaphor: {len(agreement3['no_metaphor'])} \n {[a.id for a in agreement3['no_metaphor']]}")
        print(
            f"annotator 3 and 2 disagreement: {len(disagreement3)} \n {[d.id for d in disagreement3]}")

        print(f"agreed metaphors from all annotators: {len(agreement1['is_metaphor'].intersection(agreement2['is_metaphor'], agreement3['is_metaphor']))} \n {[a.id for a in list(agreement1['is_metaphor'].intersection(agreement2['is_metaphor'], agreement3['is_metaphor']))]}")
        print(
            f"agreed no metaphors from all annotators: {len(agreement1['no_metaphor'].intersection(agreement2['no_metaphor'], agreement3['no_metaphor']))} \n {[a.id for a in list(agreement1['no_metaphor'].intersection(agreement2['no_metaphor'], agreement3['no_metaphor']))]}")

        ac.update_annotation_collection()
    else:
        print("Please set valid arguments.")


if __name__ == "__main__":
    parser = argparser.parse()
    main_data_handling(parser.parse_args())


