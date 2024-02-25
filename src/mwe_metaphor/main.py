from src.config import get_settings
from src.mwe_metaphor.controller.bert_training_controller import BertTrainingController
from src.mwe_metaphor.controller.crf_controller import CRFController
from src.mwe_metaphor.controller.gcn_training_controller import BERTWithGCNTrainingController
from src.mwe_metaphor.controller.prediction_controller import PredictionController
from src.mwe_metaphor.utils import argparser
from src.utils.database import init_db


def classification_main(arguments):
    env = get_settings()
    if env.init_db:
        init_db()
    print(f"+++ classification modus: {env.modus.value} +++")
    if arguments.bert_gnc != 0:
        print("+++ train BERT with GNC and predict +++")
        training_controller = BERTWithGCNTrainingController(settings=env)
        results = training_controller.training()

        print('K-fold cross-validation results:')
        print("Accuracy: {}".format(sum([i for i, j, k, l in results]) / env.K))
        print("Precision: {}".format(sum([j for i, j, k, l in results]) / env.K))
        print("Recall: {}".format(sum([k for i, j, k, l in results]) / env.K))
        print("F-score: {}".format(sum([l for i, j, k, l in results]) / env.K))

        # sanity checks
        print('####')
        print('recorded_results_per_fold=', results)
        print('len(set(recorded_results_per_fold))=', len(set(results)))

    elif arguments.distilBert_finetuned != 0:
        if arguments.training:
            print("+++ train DistilBERT +++")
            bert_training_controller = BertTrainingController(settings=env)
            bert_training_controller.training()

        print(f"+++ start prediction with fine-tuned model +++")
        prediction_controller = PredictionController(settings=env, pre_training=True)
        results = prediction_controller.predict()
        print(f"+++ finish prediction with results: {results} +++")

    elif arguments.distilBert_baseline != 0:
        print("+++ start prediction without fine-tuned model +++")
        prediction_controller = PredictionController(settings=env, pre_training=False)
        prediction_controller.predict()

    elif arguments.crf != 0:
        print("+++ build crf model and start prediction +++")
        crf = CRFController(settings=get_settings())
        crf.build_crf()


if __name__ == '__main__':
    parser = argparser.parse()
    classification_main(parser.parse_args())
