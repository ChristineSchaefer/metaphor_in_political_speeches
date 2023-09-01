from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.config import Settings
from src.mwe_metaphor.utils.training_utils import adjacency


class TrainingController(BaseModel):
    settings: Settings
    max_grad_norm: int = 1.0
    max_len: int = 0

    def prepare_data(self):
        df = pd.read_csv(self.settings.metaphor_dir, header=0, sep=',')
        # Create sentence and label lists
        sentences = df.sentence.values
        self.max_len = max([len(sent.split()) for sent in sentences]) + 2
        print('MAX_LEN =', self.max_len)

        A = np.array(adjacency(sentences=sentences, max_len=self.settings.max_len))

        with open(self.settings.mwe_dir) as f:
            A_MWE = mwe_adjacency(f, self.settings.metaphor_dir, self.settings.max_len - 2)
