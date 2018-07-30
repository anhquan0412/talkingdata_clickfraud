# TalkingData Kaggle competition ([link](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection#description))

This repository includes data preprocessing, feature engineering and machine learning techniques to produce the top 13% results on Kaggle Private leaderboard

## Preprocess
- [Feature Engineering](https://github.com/anhquan0412/talkingdata_clickfraud/blob/master/new_fe.ipynb) : calculate time to next click/previous click, features' cummulative counts, lag features ... 
- [Mean Encoding Features + Generate train/validation/test set features](https://github.com/anhquan0412/talkingdata_clickfraud/blob/master/mean_encoding_and_val_merge.ipynb)

## Modeling
- [Random Forest](https://github.com/anhquan0412/talkingdata_clickfraud/blob/master/new_RF.ipynb) Hyperparameter tuning, evaluate Random Forest's feature importance and visualize redundant features using dendogram
- [XGBoost](https://github.com/anhquan0412/talkingdata_clickfraud/blob/master/xgboost.ipynb) Hyperparameter tuning (without tree depth tuning), visualize feature importance. This notebook produces final submission file.
- [Deep Neural Network with categorical embeddings](https://github.com/anhquan0412/talkingdata_clickfraud/blob/master/fastai.ipynb) Built with pytorch and fast.ai library wrapper. Use cyclical learning rate to speed up training process.
- [Blending](https://github.com/anhquan0412/talkingdata_clickfraud/blob/master/fastai.ipynb) Simple average blending and short tutorial of using numpy memory map to save memory while process data.
