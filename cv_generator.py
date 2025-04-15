import numpy as np
import pandas as pd

class KalmanCV:
    def __init__(self, model_class, model_config, window_size=60, holdout=20):
        self.model_class = model_class  
        self.model_config = model_config
        self.window_size = window_size
        self.holdout = holdout
        self.results = []

    def run_cv(self, Y):
        n_obs = Y.shape[0]
        n_splits = n_obs - self.window_size - self.holdout + 1

        for start in range(n_splits):
            train_start = start
            train_end = start + self.window_size
            val_start = train_end
            val_end = train_end + self.holdout

            Y_train = Y[train_start:train_end]
            Y_val = Y[val_start:val_end]

            model = self.model_class(self.model_config)
            train_loss = model.fit(Y_train)
            val_loss = model.evaluate(Y_val)

            self.results.append({
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        return pd.DataFrame(self.results)