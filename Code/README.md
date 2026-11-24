## Code 

Code folder for main project code.

| File | Function |
| ------- | ------- |
| dataloader.py | Loads model data from gitignored foler in "data." Includes dataset's recommended split for preventing data leakage in "split_by_fold" function |
| metrics.py | Evaluation metrics for models such as ROC-AUC and PR-AUC | 
| ml_baselines.py | ML model baselines on extracted features to compare model performance versus more classic ML models on our data. | 
| model.py | 1D-CNN and LSTM model classes | 
| train.py | Model training script, GPU use if availible, augmentations/transforms, runs model training, saving checkpoint. |
