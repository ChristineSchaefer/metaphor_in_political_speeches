import torch

from src.config import get_settings
from src.mwe_metaphor.controller.training_controller import TrainingController


def main():
    env = get_settings()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_controller = TrainingController(settings=env)
    training_controller.prepare_data()


if __name__ == '__main__':
    main()
