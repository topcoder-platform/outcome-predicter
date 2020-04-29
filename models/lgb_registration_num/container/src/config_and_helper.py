TRAINING_FILE = 'train.csv'

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import f1_score
import lightgbm as lgb
import hyperopt
import boto3
import subprocess
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
#from tsfresh.feature_extraction import feature_calculators
import time
import os
import datetime
import math
import pickle

# s3 resource for storing cached pickle files
s3_resource = boto3.Session().resource('s3')

def class_binaryzation(input_data, target_columns=['Challenge Manager', 'Challenge Copilot', 'Track', 'Technology List']):
    """
    Function to binaryze class data
    """
    output = pd.DataFrame()
    extended_target_columns = {}
    column_with_0 = [0 for _ in range(len(input_data))]

    for column in input_data.columns:
        if column not in target_columns:
            output[column] = input_data[column]
        else:
            extended_target_columns[column] = []
            for i in range(len(input_data)):
                values = input_data.loc[i, column].split(',')
                for value in values:
                    if value not in output.columns:
                        output[value] = column_with_0.copy()
                        extended_target_columns[column].append(value)
                    output.loc[i, value] = 1
    
    return output, extended_target_columns

def class_binaryzation_for_test(input_data, extended_target_columns, 
                       target_columns=['Challenge Manager', 'Challenge Copilot', 'Track', 'Technology List']):
    """
	Function to binaryze class data
    """
    output = pd.DataFrame()
    column_with_0 = [0 for _ in range(len(input_data))]

    for column in input_data.columns:
        if column not in target_columns:
            output[column] = input_data[column]
        else:
            for extended_target_column in extended_target_columns[column]:
                output[extended_target_column] = column_with_0.copy()
            for i in range(len(input_data)):
                if isinstance(input_data.loc[i, column], str):
                    values = input_data.loc[i, column].split(',')
                    for value in values:
                        if value in output.columns:
                            output.loc[i, value] = 1
                        else:
                            print('data error, new value: '+value)
                elif not math.isnan(input_data.loc[i, column]):
                    print('data error')
                    exit(1)
         
    return output

def date_separation1(input_data, target_columns=['Submitby Date Time', 'Posting Date Date'], max_num_columns=6):
    """
	Function to separate date data v1
    """
    target_suffixes = [' Year', ' Month', ' Day', ' Hour', ' Minute', 'Second']
    output = pd.DataFrame()
    column_with_none = [None for _ in range(len(input_data))]

    for column in input_data.columns:
        if column not in target_columns:
            output[column] = input_data[column]
        else:
            for i in range(len(input_data)):
                values = input_data.loc[i, column].replace(' ', '-').replace(':', '-').split('-')
                if len(values) not in [3, 6]:
                    print('data error')
                    print('len(values)='+str(len(values)))
                    print('i='+str(i))
                    exit(1)
                for j in range(min(len(values), max_num_columns)):
                    if (column+target_suffixes[j]) not in output.columns:
                        output[column+target_suffixes[j]] = column_with_none.copy()
                    output.loc[i, column+target_suffixes[j]] = int(values[j])

    return output

def date_separation2(input_data, target_columns=['Submitby Date Time', 'Posting Date Date']):
    """
    Function to separate date data v2
    """
    target_suffixes = [' Day', ' Month', ' Year', ' Hour', ' Minute']
    output = pd.DataFrame()
    column_with_none = [None for _ in range(len(input_data))]

    for column in input_data.columns:
        if column not in target_columns:
            output[column] = input_data[column]
        else:
            for i in range(len(input_data)):
                values = input_data.loc[i, column].replace(' ', '-').replace(':', '-').replace('/', '-').split('-')
                if len(values) not in [3, 5]:
                    print('data error')
                    exit(1)
                for j in range(len(values)):
                    if (column+target_suffixes[j]) not in output.columns:
                        output[column+target_suffixes[j]] = column_with_none.copy()
                    output.loc[i, column+target_suffixes[j]] = int(values[j])

    return output

def money_digitalization(raw_data, target_columns=['First Place Prize', 'Total Prize']):
    """
	Function to digitalize prize data
    """
    output = raw_data.copy()

    for column in target_columns:
        for i in range(len(raw_data)):
            money = raw_data.loc[i, column].replace('(', '').replace(')', '')
            if money[0] == '$':
                output.loc[i, column] = float(money[1:].replace('.', '').replace(',', '.'))
            elif money[:3] == 'US$':
                output.loc[i, column] = float(money[3:].replace('.', '').replace(',', '.'))
            else:
                print('money data error')
                exit(1)

    return output

def get_date_in_days(raw_data, target_columns=['Submitby Date Time', 'Posting Date Date']):
    """
    Function to calculate days from 2016.1.1
    """
    output = raw_data.copy()

    for column in target_columns:
        for i in range(len(raw_data)):
            date = datetime.date(output.loc[i, column+' Year'], output.loc[i, column+' Month'], output.loc[i, column+' Day'])
            output.loc[i, column+' Days from 2016'] = (date-datetime.date(2016, 1, 1)).days

    return output

def training_data_preprocessing(raw_data, num_passed_rows=72):
    """
    Function to organize data
    input:
        raw_data: DataFrame of raw data 
    output:
        data_output: DataFrame of organized data
        label_output: DataFrame of labels
    """
    # some samples have errors
    raw_data = raw_data[num_passed_rows:].reset_index(drop=True) 
    
    # get data output
    data_output = raw_data[['Submitby Date Time', 'Challenge Manager', 'Challenge Copilot', 'Posting Date Date', 'Track',
                            'Technology List', 'First Place Prize', 'Num Registrations', 'Total Prize']]
    data_output, extended_columns = class_binaryzation(data_output)
    
    # save extended columns to cache
    extended_columns_filepath = 'cache/extended_columns.pkl'
    with open(extended_columns_filepath, 'wb') as f:
        pickle.dump(extended_columns, f)

    num_date_columns_filepath = 'cache/num_date_columns.pkl'
    try:
        data_output = date_separation1(data_output) 
        with open(num_date_columns_filepath, 'wb') as f:
            pickle.dump(6, f)

    except:
        data_output = date_separation2(data_output)
        with open(num_date_columns_filepath, 'wb') as f:
            pickle.dump(5, f)

    data_output = money_digitalization(data_output)
    data_output = get_date_in_days(data_output)
    data_output['Days from Posting to Submit'] = data_output['Submitby Date Time Days from 2016'] \
                                                 - data_output['Posting Date Date Days from 2016'] 
    
    # get other output
    label_output = pd.DataFrame(columns=['Success'])
    success_output = pd.DataFrame(columns=data_output.columns)
    failure_output = pd.DataFrame(columns=data_output.columns)
    for i in range(len(raw_data)):
        if raw_data.loc[i, 'Num Submissions Passed Review'] >= 1:
            label_output.loc[i, 'Success'] = 1
            success_output.loc[len(success_output)] = data_output.loc[i]
        else:
            label_output.loc[i, 'Success'] = 0
            failure_output.loc[len(failure_output)] = data_output.loc[i]

    return data_output, label_output, success_output, failure_output, extended_columns

def test_data_preprocessing(raw_data):
    """
    Function to organize data
    input:
        raw_data: DataFrame of raw data 
    output:
        data_output: DataFrame of organized data
        label_output: DataFrame of labels
    """ 

    # get data output
    data_output = raw_data[['Submitby Date Time', 'Challenge Manager', 'Challenge Copilot', 'Posting Date Date', 'Track',
                            'Technology List', 'First Place Prize', 'Num Registrations', 'Total Prize']]
    with open('cache/extended_columns.pkl', 'rb') as f:
        extended_columns = pickle.load(f)
    with open('cache/num_date_columns.pkl', 'rb') as f:
        max_date_columns = pickle.load(f)
    
    data_output = class_binaryzation_for_test(data_output, extended_columns)
    try:
        data_output = date_separation1(data_output, max_num_columns=NUM_DATE_COLUMNS)
    except:
        data_output = date_separation2(data_output)
    data_output = money_digitalization(data_output)
    data_output = get_date_in_days(data_output)
    data_output['Days from Posting to Submit'] = data_output['Submitby Date Time Days from 2016'] \
                                                 - data_output['Posting Date Date Days from 2016'] 

    return data_output


def lgb_hyperopt(data, labels, num_evals=1000, n_folds=5, diagnostic=False):
    """
    Function to turn parameters for Lightgbm
    """
    LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
    LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM 
    EVAL_METRIC_LGBM_CLASS = 'f1'

    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return 'f1', f1_score(y_true, y_hat), True

    print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))
    #clear space
        
    integer_params = ['max_depth',
                        'num_leaves',
                        'max_bin',
                        'min_data_in_leaf',
                        'min_data_in_bin']
        
    def objective(space_params):
            
        #cast integer params from float to int
        for param in integer_params:
            space_params[param] = int(space_params[param])
            
        #extract nested conditional parameters
        if space_params['boosting']['boosting'] == 'goss':
            top_rate = space_params['boosting'].get('top_rate')
            other_rate = space_params['boosting'].get('other_rate')
            #0 <= top_rate + other_rate <= 1
            top_rate = max(top_rate, 0)
            top_rate = min(top_rate, 0.5)
            other_rate = max(other_rate, 0)
            other_rate = min(other_rate, 0.5)
            space_params['top_rate'] = top_rate
            space_params['other_rate'] = other_rate
            
        subsample = space_params['boosting'].get('subsample', 1.0)
        space_params['boosting'] = space_params['boosting']['boosting']
        space_params['subsample'] = subsample
            
        cv_results = lgb.cv(space_params, train, nfold = n_folds, stratified=True,
                            early_stopping_rounds=100, seed=42, feval=lgb_f1_score)
            
        best_loss = -cv_results['f1-mean'][-1]

        return{'loss':best_loss, 'status': STATUS_OK }
        
    train = lgb.Dataset(data, labels)
                
    #integer and string parameters, used with hp.choice()
    boosting_list = [{'boosting': 'gbdt',
                      'subsample': hp.uniform('subsample', 0.5, 1)},
                      {'boosting': 'goss',
                       'subsample': 1.0,
                       'top_rate': hp.uniform('top_rate', 0, 0.5),
                       'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'

    objective_list_reg = ['huber', 'gamma', 'fair', 'tweedie']
    objective_list_class = ['binary', 'cross_entropy']
    objective_list = objective_list_class
    is_unbalance_list = [True]

    space ={'boosting' : hp.choice('boosting', boosting_list),
            'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
            'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
            'max_bin': hp.quniform('max_bin', 32, 255, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
            'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),
            'min_gain_to_split' : hp.quniform('min_gain_to_split', 0.1, 5, 0.01),
            'lambda_l1' : hp.uniform('lambda_l1', 0, 5),
            'lambda_l2' : hp.uniform('lambda_l2', 0, 5),
            'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
            'metric' : None, 
            'objective' : hp.choice('objective', objective_list),
            'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),
            'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01),
            'is_unbalance' : hp.choice('is_unbalance', is_unbalance_list)
        }

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=num_evals, 
                trials=trials)
                
    #fmin() will return the index of values chosen from the lists/arrays in 'space'
    #to obtain actual values, index values are used to subset the original lists/arrays
    #extract nested conditional parameters
    try:
        if best['boosting']['boosting'] == 'goss':
            top_rate = best['boosting'].get('top_rate')
            other_rate = best['boosting'].get('other_rate')
            #0 <= top_rate + other_rate <= 1
            top_rate = max(top_rate, 0)
            top_rate = min(top_rate, 0.5)
            other_rate = max(other_rate, 0)
            other_rate = min(other_rate, 0.5)
            best['top_rate'] = top_rate
            best['other_rate'] = other_rate
    except:
        if boosting_list[best['boosting']]['boosting'] == 'goss':
            top_rate = best['top_rate']
            other_rate = best['other_rate']
            #0 <= top_rate + other_rate <= 1
            top_rate = max(top_rate, 0)
            top_rate = min(top_rate, 0.5)
            other_rate = max(other_rate, 0)
            other_rate = min(other_rate, 0.5)
            best['top_rate'] = top_rate
            best['other_rate'] = other_rate
    best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice
    best['metric'] = metric_list[best['metric']]
    best['objective'] = objective_list[best['objective']]
    best['is_unbalance'] = is_unbalance_list[best['is_unbalance']]
                
    #cast floats of integer params to int
    for param in integer_params:
        best[param] = int(best[param])
        
    print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
    if diagnostic:
        return(best, trials)
    else:
        return(best)

def train_model(X_train, y_train, X_valid, y_valid, params=None, model_type='lgb', 
                model_path_name='lgb', plot_feature_importance=False, model=None):
    """
    Function to train a model
    """
    def lgb_f1_score(y_true, y_pred):
        y_pred = np.round(y_pred)
        return 'f1', f1_score(y_true, y_pred), True

    scores = []
    feature_importance = pd.DataFrame()
    print('Started at', time.ctime())
    
        
    if model_type == 'lgb':
        
        model = lgb.LGBMClassifier(**params, n_estimators=50000, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), 
                    eval_metric=lgb_f1_score, early_stopping_rounds=300)
            
        y_pred_valid = model.predict(X_valid)
        
    if model_type == 'cat':
        model = cb.CatBoost(iterations=20000, **params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
        y_pred_valid = model.predict(X_valid)

    #save the model
    joblib.dump(model, model_path_name)
     
    scores.append(f1_score(y_valid, y_pred_valid)) 
        
    if model_type == 'lgb':
        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = X_train.columns
        fold_importance["importance"] = model.feature_importances_
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
    print('score: {0:.4f}.'.format(np.mean(scores)))

    if model_type == 'lgb':
        feature_importance["importance"]
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            #sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
        
            return feature_importance, np.mean(scores)
        return np.mean(scores)
    
    else:
        return np.mean(scores)

def calculate_f1(dataframe_true, dataframe_pred):

    return f1_score(1-dataframe_true['Success'].values, 1-np.round(dataframe_pred['Success Probability'].values))

def get_label(raw_data):
    """
    Function to get labels for a dataframe
    """
    label_output = pd.DataFrame(columns=['Success'])
    for i in range(len(raw_data)):
        if raw_data.loc[i, 'Num Submissions Passed Review'] >= 1:
            label_output.loc[i, 'Success'] = 1
        else:
            label_output.loc[i, 'Success'] = 0

    return label_output

def copy_files(src, desc, file_list):
    """
    Function to copy files to desc directory.
    """
    src_paths = [os.path.join(src, file_name) for file_name in file_list]
    output = subprocess.run(['cp', '-a', *src_paths, desc], check=True)
    if output.returncode != 0:
        raise Exception('Failed to copy files from {} to {}'.format(src, desc))
