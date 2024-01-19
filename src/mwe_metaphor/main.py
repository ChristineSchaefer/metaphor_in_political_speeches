from src.config import get_settings
from src.mwe_metaphor.controller.bert_training_controller import BertTrainingController
from src.mwe_metaphor.controller.gcn_training_controller import TrainingController
from src.mwe_metaphor.controller.prediction_controller import PredictionController
from src.mwe_metaphor.utils import argparser


def classification_main(arguments):
    env = get_settings()
    if arguments.bert_gnc != 0:
        print("+++ train BERT with GNC and predict +++")
        training_controller = TrainingController(settings=env)
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

    elif arguments.distilBert_finetuned != 0 and arguments.num_epochs != 0:
        if arguments.training:
            print("+++ train DistilBERT +++")
            bert_training_controller = BertTrainingController(settings=env)
            bert_training_controller.training()

        print("+++ start prediction with fine-tuned model +++")
        prediction_controller = PredictionController(settings=env, pre_training=True, num_epochs=arguments.num_epochs)
        prediction_controller.predict()

    elif arguments.distilBert_finetuned != 0 and arguments.num_epochs == 0:
        if arguments.training:
            print("+++ train DistilBERT +++")
            bert_training_controller = BertTrainingController(settings=env)
            bert_training_controller.training()

        print("+++ start prediction with fine-tuned model +++")
        prediction_controller = PredictionController(settings=env, pre_training=True)
        prediction_controller.predict()

    elif arguments.distilBert_baseline != 0:
        print("+++ start prediction without fine-tuned model +++")
        prediction_controller = PredictionController(settings=env, pre_training=False)
        prediction_controller.predict()


if __name__ == '__main__':
    parser = argparser.parse()
    classification_main(parser.parse_args())
