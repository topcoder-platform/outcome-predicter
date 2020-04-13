## A model on new challenge data 
**Introduction**
This is a first try of modeling the new challenge data. Only one model 'lightbgm' is used. All registrant features are vectorized (numerical features are vectorized with intervals) and used as statistic data of each challenge.
You can follow the content below to see the model's result.

**Prerequisites**
- Python 3.6
- Pip

**Dependencies installation**
Install the required dependencies by executing the command:
```
pip install -r requirements.txt
```

**Training on 'training_data/data.csv'**
Run the command:
```
python train.py ../../training_data/data.csv lgb
```
It will take a while for data preprocessing.

**Prediction on 'test_data/test.csv'**
Run the command:
```
python predict.py lgb ../../test_data/test.csv 
```
Note that default output is 'predictions.csv'.

## Improvement to be implemented
Since for a certain challenge, we can use all the data (features) of already finished challenges, a lot of new registrant features such as registration counts, submission counts, win counts topcoder ages and earning counts at that time can be mined and utilized.
Currently only one kind of statistic features (mean values) are used, other statistic features are to be tried, and we can delete some trivial features for better generalization.
Other models like 'xgboost', 'catboost', neural networks and traditional SVM are to be adopted, and also we can combine multiple models after discovered features are enough.

## Others
After training, feature importance will be generated in 'other_resources/'.
The nodebook of parameter turning is in 'other_resources/', but since a set of turned parameters have been set, I believe it should be considered only when training data changes dramatically.

