from src.config import get_settings
from src.mwe_metaphor.controller.gcn_training_controller import TrainingController


def main():
    env = get_settings()
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


if __name__ == '__main__':
    main()
