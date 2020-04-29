TRAINING_FILE = 'train.csv'

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import f1_score
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
#from tsfresh.feature_extraction import feature_calculators
import time
import datetime
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import math

to_be_extended_columns={'class':['Challenge Stats Project Category Name', 'Challenge Stats Tco Track',
                                 'Challenge Stats Challenge Manager', 'Challenge Stats Challenge Copilot'
                                 'Challenge Stats Track', 'Challenge Stats Technology List', 
                                 'Member Profile Advanced Reporting Country'],
                        'number':['User Member Since Date Year', 'User Member Since Date Month', 
                                  'User Member Since Date Day', 'User Member Since Date Days from 2001', 
                                  'Challenge Stats Old Rating']}

def class_binaryzation(input_data, target_columns=['Challenge Stats Project Category Name', 'Challenge Stats Tco Track',
                                                  'Challenge Stats Challenge Manager', 'Challenge Stats Challenge Copilot',
                                                  'Challenge Stats Track', 'Challenge Stats Technology List', 
                                                  'Member Profile Advanced Reporting Country'],
                       extended_target_columns=None):
    """
    Function to binaryze class data
    """
    print('start class binaryzation')

    output = pd.DataFrame()
    column_with_0 = [0 for _ in range(len(input_data))]
    if extended_target_columns == None:
        get_extended_columns = True
        extended_target_columns = {}
    else:
        get_extended_columns = False

    for column in input_data.columns:
        if column not in target_columns:
            output[column] = input_data[column]
        else:
            print('binaryzation:'+column)
            if get_extended_columns:
                extended_target_columns[column] = []
            else:
                for extended_target_column in extended_target_columns[column]:
                    output[extended_target_column] = column_with_0.copy()
            for i in range(len(input_data)):
                values = input_data.at[i, column]
                try:
                    if isinstance(values, float) and math.isnan(values):
                        continue
                    else:
                        values = input_data.at[i, column].split(',')
                except:
                    print('error, value:')
                    print(input_data.at[i, column])
                    print('at '+str(i)+' '+column)
                    exit(1)
                for value in values:
                    if ((column+':'+value) not in output.columns):
                        if not get_extended_columns:
                            continue
                        output[column+':'+value] = column_with_0.copy()
                        extended_target_columns[column].append(column+':'+value)
                    output.at[i, column+':'+value] = 1
    
    return output, extended_target_columns

def parse_interval(input):
    """
    Function to parse an interval from a string
    """
    output = input.replace(']', '[').replace('(', '[').replace(')', '[').replace(',', '[').split('[')[-3:-1]
    if output[0] == 'lower':
        output[0] = -99999999999
    else:
        output[0] = float(output[0])
    if output[1] == 'higher':
        output[1] = 99999999999
    else:
        output[1] = float(output[1])

    return output

def number_binaryzation(input_data, target_columns=['User Member Since Date Year', 'User Member Since Date Month', 
                                                    'User Member Since Date Day', 'User Member Since Date Days from 2001', 
                                                    'Challenge Stats Old Rating'],
                        nums_intervals=[None, None, 15, 15, 15], extended_target_columns=None):
    """
    Function to binaryze number data (nums_intervals = nums_intervals - 1)
    """    
    print('start number binaryzation')

    output = pd.DataFrame()
    interval_mapping = {}
    column_with_0 = [0 for _ in range(len(input_data))]
    if extended_target_columns == None:
        get_extended_columns = True
        extended_target_columns = {}
    else:
        get_extended_columns = False
    
    for column in input_data.columns:
        if column not in target_columns:
            output[column] = input_data[column]
        else:
            interval_mapping[column] = []
            column_numbers = input_data.loc[:, column].values.tolist()
            column_numbers = [value for value in column_numbers if value!=None]
            min_column_number = min(column_numbers)
            max_column_number = max(column_numbers)
            difference = max_column_number - min_column_number
            if nums_intervals[target_columns.index(column)] == None:
                nums_intervals[target_columns.index(column)] = int(difference)
            num_intervals = nums_intervals[target_columns.index(column)]
            if num_intervals == 0: 
                step = 0
            else:
                step = difference / num_intervals
            num_intervals += 1

            if get_extended_columns:
                extended_target_columns[column] = []
                for i in range(num_intervals):
                    boundary1 = min_column_number + (i-0.5)*step
                    boundary2 = min_column_number + (i+0.5)*step
                    if i == 0:     
                        new_column = column +':(' + 'lower' + ',' + str(boundary2) + ')'
                    elif i == num_intervals-1:
                        new_column = column +':[' + str(boundary1) + ',' + 'higher' + ')'
                    else:        
                        new_column = column + ':[' + str(boundary1) + ',' + str(boundary2) + ')'
                    extended_target_columns[column].append(new_column)
                    output[new_column] = column_with_0.copy()
                    interval_mapping[column].append([boundary1, boundary2])
            else:
                for extended_column in extended_target_columns[column]:
                    interval_mapping[column].append(parse_interval(extended_column))
                    output[extended_column] = column_with_0.copy()
            
            for i in range(len(input_data)):
                #try:
                if input_data.at[i, column] != None:
                    value = float(input_data.at[i, column])
                    for j in range(num_intervals):
                        if value < interval_mapping[column][j][1]:
                            output.at[i, extended_target_columns[column][j]] = 1
                            break
                        if j == num_intervals:
                            print('place error at number_binaryzation')
                            exit(1)
                else:
                    print('missing value:')
                    print(input_data.at[i, column])
    
    return output, extended_target_columns

def feature_processing(input_df):
    """
    Function to get features for a challenge
    """
    return np.mean(input_df.values, axis=0)

def data_merging(input_data, id_column='Challenge Stats Challenge ID', label_column='Challenge Stats Status Desc', 
                 merged_columns=['Member Profile Advanced Reporting Country', 'User Member Since Date', 
                                 'Challenge Stats Old Rating']):
    """
    Function to merging member data
    """
    print('start data merging')

    output = []
    used_input_data = input_data.copy().sort_values(by=[id_column]).reset_index(drop=True)
    merged_column_i_list = []
    input_column_list = list(input_data.columns)
    for column in merged_columns:
        merged_column_i_list.append(input_column_list.index(column))

    challenge_id = used_input_data.at[0, id_column]
    cache_i_start = 0
    for i, row in used_input_data.iterrows():
        if challenge_id != row[id_column]:
            data_cache = used_input_data[cache_i_start:i]
            output_row_data = data_cache.values[0]
            output_row_data[merged_column_i_list] = feature_processing(data_cache[merged_columns])
            output.append(output_row_data)
            challenge_id = row[id_column]
            cache_i_start = i
            if i%50 == 0:
                print(str(i+1)+' passed')

    # last processing
    data_cache = used_input_data[cache_i_start:]
    output_row_data = data_cache.values[0]
    output_row_data[merged_column_i_list] = feature_processing(data_cache[merged_columns])
    output.append(output_row_data)  
    
    return pd.DataFrame(output, columns=input_data.columns)

def date_separation(input_data, target_columns=['Challenge Stats Submitby Date Date', 'Challenge Stats Posting Date Date',
                                                'User Member Since Date']):
    """
    Function to separate date data
    """
    print('start date separation')

    target_suffixes = [' Year', ' Month', ' Day']
    output = pd.DataFrame()
    column_with_none = [None for _ in range(len(input_data))]

    for column in input_data.columns:
        if column not in target_columns:
            output[column] = input_data[column]
        else:
            for i in range(len(input_data)):
                values = input_data.at[i, column]
                if isinstance(values, str):
                    values = [int(value) for value in values.replace(' ', '-').replace(':', '-').replace('/', '-').split('-')][:3]
                elif isinstance(values, pd.Timestamp):
                    values = [values.year, values.month, values.day]
                else:
                    print('found missing date:')
                    print(values)
                    print('type: '+str(type(values)))
                    print('')
                    values = [None, None, None]
                for j in range(len(values)):
                    if (column+target_suffixes[j]) not in output.columns:
                        output[column+target_suffixes[j]] = column_with_none.copy()
                    output.at[i, column+target_suffixes[j]] = values[j]

    return output

def get_date_in_days(raw_data, target_columns=['Challenge Stats Submitby Date Date', 'Challenge Stats Posting Date Date',
                                               'User Member Since Date']):
    """
    Function to calculate days from 2001.1.1
    
    """
    print('start getting date in days')

    output = raw_data.copy()

    for column in target_columns:
        for i in range(len(raw_data)):
            year = output.at[i, column+' Year']
            month = output.at[i, column+' Month']
            day = output.at[i, column+' Day']
            if (year==None) or (month==None) or (day==None):
                output.at[i, column+' Days from 2001'] = None
            else:
                date = datetime.date(year, month, day)
                output.at[i, column+' Days from 2001'] = (date-datetime.date(2001, 1, 1)).days

    return output

def column_analysis(input_data, to_be_extended_columns, nums_intervals=[None, None, 15, 15, 15]):
    """
    Fuction to get names of extended target columns
    """
    extended_columns = {'class':{}, 'number':{}}
    column_with_0 = [0 for _ in range(len(input_data))]

    for column in input_data.columns:
        # class
        if column in to_be_extended_columns['class']:        
            extended_columns['class'][column] = []
            for i in range(len(input_data)):
                values = input_data.loc[i, column]
                try:
                    if isinstance(values, float) and math.isnan(values):
                        values = ['unknown']
                    else:
                        values = input_data.loc[i, column].split(',')
                except:
                    print('error, value:')
                    print(input_data.loc[i, column])
                    print('at '+str(i)+' '+column)
                    exit(1)
                for value in values:
                    if ((column+':'+value) not in extended_columns['class'][column]):
                        extended_columns['class'][column].append(column+':'+value)
        # number
        elif column in to_be_extended_columns['number']:
            extended_columns['number'][column] = []
            column_numbers = input_data.loc[:, column].values.tolist()
            column_numbers = [value for value in column_numbers if value!=None]
            min_column_number = min(column_numbers)
            max_column_number = max(column_numbers)
            difference = max_column_number - min_column_number
            if nums_intervals[to_be_extended_columns['number'].index(column)] == None:
                nums_intervals[to_be_extended_columns['number'].index(column)] = int(difference)
            
            num_intervals = nums_intervals[to_be_extended_columns['number'].index(column)]
            if num_intervals == 0:
                step = 0
            else:
                step = difference / num_intervals
            for i in range(num_intervals):
                boundary1 = min_column_number + (i-0.5)*step
                boundary2 = min_column_number + (i+0.5)*step
                if i == 0:     
                    new_column = column +':(' + 'lower' + ',' + str(boundary2) + ')'
                elif i == num_intervals-1:
                    new_column = column +':[' + str(boundary1) + ',' + 'higher' + ')'
                else:
                    boundary2 = min(column_numbers) + (i+1)*step         
                    new_column = column + ':[' + str(boundary1) + ',' + str(boundary2) + ')'
                extended_columns['number'][column].append(new_column)

    return extended_columns

def training_data_preprocessing(raw_data, extended_columns={'class':None, 'number':None}):
    """
    Function to organize data
    input:
        raw_data: DataFrame of raw data 
        extended_columns: dict of extended columns
    output:
        data_output: DataFrame of organized data
        label_output: DataFrame of labels
    """
    # do frist screening
    data_output = raw_data[raw_data['Challenge Stats Status Desc'].isin(
        ['Completed', 'Cancelled - Zero Submissions', 'Cancelled - Failed Review'])].reset_index(drop=True)
    print('rows_left: '+str(len(data_output)))
    data_output = data_output[['Challenge Stats Project Category Name', 'Challenge Stats Submitby Date Date',
                               'Challenge Stats Tco Track', 'Challenge Stats Challenge Manager',
                               'Challenge Stats Challenge Copilot', 'Challenge Stats Posting Date Date',
                               'Challenge Stats Track', 'Challenge Stats Technology List', 'Challenge Stats First Place Prize',
                               'Challenge Stats Total Prize', 'Challenge Stats Num Registrations', 
                               'Member Profile Advanced Reporting Country', #'Challenge Stats Registrant Handle', 
                               'User Member Since Date', 'Challenge Stats Old Rating',
                               # used for later processing
                               'Challenge Stats Challenge ID', 'Challenge Stats Status Desc']]

    # get data output
    data_output, extended_class_columns = class_binaryzation(data_output,extended_target_columns=extended_columns['class'])
    data_output = date_separation(data_output)
    data_output = get_date_in_days(data_output)
    data_output, extended_number_columns = number_binaryzation(data_output, extended_target_columns=extended_columns['number'])

    merged_columns = extended_class_columns['Member Profile Advanced Reporting Country'].copy()
    for index in extended_number_columns:
        merged_columns += extended_number_columns[index]
    data_output = data_merging(data_output, merged_columns=merged_columns)
    print(data_output['Challenge Stats Challenge ID'])
    data_output['Days from Posting to Submit'] = data_output['Challenge Stats Submitby Date Date Days from 2001'] \
                                                 - data_output['Challenge Stats Posting Date Date Days from 2001'] 
    
    # get other output
    print('start to get other output')
    success_output = data_output[data_output['Challenge Stats Status Desc']\
        .isin(['Completed'])].reset_index(drop=True)
    failure_output = data_output[data_output['Challenge Stats Status Desc']\
        .isin(['Cancelled - Zero Submissions', 'Cancelled - Failed Review'])].reset_index(drop=True)

    label_output = pd.DataFrame(columns=['Success'])
    label_output['Success'] = data_output['Challenge Stats Status Desc']
    def applied_func(row):
        if row[0] == 'Completed':
            return 1
        return 0
    label_output = label_output.apply(applied_func, axis=1, result_type='broadcast')
    print(label_output)

    # drop unuseful columns
    data_output = data_output.drop(columns=['Challenge Stats Challenge ID', 'Challenge Stats Status Desc'])
    success_output = success_output.drop(columns=['Challenge Stats Challenge ID', 'Challenge Stats Status Desc'])
    failure_output = failure_output.drop(columns=['Challenge Stats Challenge ID', 'Challenge Stats Status Desc'])

    extended_columns = {'class': extended_class_columns, 'number':extended_number_columns}
    return data_output, label_output, success_output, failure_output, extended_columns

def get_label(raw_data):
    """
    Function to get labels for a dataframe
    """
    label_output = pd.DataFrame(columns=['Success'])
    label_output['Success'] = raw_data['Challenge Stats Status Desc']
    def applied_func(row):
        if row[0] == 'Completed':
            return 1
        return 0
    label_output = label_output.apply(applied_func, axis=1, result_type='broadcast')

    return label_output

def test_data_preprocessing(raw_data):
    """
    Function to organize data
    input:
        raw_data: DataFrame of raw data 
    output:
        data_output: DataFrame of organized data
    """ 

    # do frist screening
    data_output = raw_data[raw_data['Challenge Stats Status Desc'].isin(
        ['Completed', 'Cancelled - Zero Submissions', 'Cancelled - Failed Review'])].reset_index(drop=True)
    print('rows_left: '+str(len(data_output)))
    data_output = data_output[['Challenge Stats Project Category Name', 'Challenge Stats Submitby Date Date',
                               'Challenge Stats Tco Track', 'Challenge Stats Challenge Manager',
                               'Challenge Stats Challenge Copilot', 'Challenge Stats Posting Date Date',
                               'Challenge Stats Track', 'Challenge Stats Technology List', 'Challenge Stats First Place Prize',
                               'Challenge Stats Total Prize', 'Challenge Stats Num Registrations', 
                               'Member Profile Advanced Reporting Country', #'Challenge Stats Registrant Handle', 
                               'User Member Since Date', 'Challenge Stats Old Rating',
                               # used for later processing
                               'Challenge Stats Challenge ID', 'Challenge Stats Status Desc']]

    # get data output
    with open('cache/extended_columns.pkl', 'rb') as f:
        extended_columns = pickle.load(f)
    data_output, extended_class_columns = class_binaryzation(data_output,extended_target_columns=extended_columns['class'])
    data_output = date_separation(data_output)
    data_output = get_date_in_days(data_output)
    data_output, extended_number_columns = number_binaryzation(data_output, extended_target_columns=extended_columns['number'])

    merged_columns = extended_class_columns['Member Profile Advanced Reporting Country'].copy()
    for index in extended_number_columns:
        merged_columns += extended_number_columns[index]
    data_output = data_merging(data_output, merged_columns=merged_columns)
    data_output['Days from Posting to Submit'] = data_output['Challenge Stats Submitby Date Date Days from 2001'] \
                                                 - data_output['Challenge Stats Posting Date Date Days from 2001'] 
    
    # cache labels
    labels = get_label(data_output)
    labels['Challenge Stats Challenge ID'] = data_output['Challenge Stats Challenge ID']
    labels.to_csv('cache/test_labels.csv', index=False)

    # drop unuseful columns
    data_output = data_output.drop(columns=['Challenge Stats Challenge ID', 'Challenge Stats Status Desc'])

    return data_output


def lgb_hyperopt(data, labels, num_evals=1000, n_folds=6, diagnostic=False):
    """
    Function to turn parameters for Lightgbm
    """
    LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
    LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM 
    EVAL_METRIC_LGBM_CLASS = 'f1'

    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)
        return 'f1', f1_score(1-y_true, 1-y_hat), True

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

    """
    def lgb_f1_score(y_true, y_pred):
        y_pred = np.round(y_pred)
        return 'f1', f1_score(1-y_true, 1-y_pred), True

    scores = []
    feature_importance = pd.DataFrame()
    print('Started at', time.ctime())
    
        
    if model_type == 'lgb':
        
        model = lgb.LGBMClassifier(**params, n_estimators=500000, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), 
                    eval_metric=lgb_f1_score, early_stopping_rounds=300)
            
        y_pred_valid = model.predict(X_valid)
        
    if model_type == 'cat':
        model = cb.CatBoost(iterations=20000, **params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
        y_pred_valid = model.predict(X_valid)

    #save the model
    joblib.dump(model, model_path_name)
     
    scores.append(f1_score(1-y_valid, 1-y_pred_valid)) 
        
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

def divide_data(raw_data, training_data_path, test_data_path, time_thr=pd.Timestamp(2020, 1, 1), num_passed_rows=962):
    """
    Function to divide raw data
    """
    raw_data = raw_data[num_passed_rows:]  # pass some rows due to data errors
    raw_test_data = raw_data[raw_data['Challenge Stats Submitby Date Date']>=time_thr].reset_index(drop=True)
    print('test row number: '+str(len(raw_test_data)))
    raw_test_data.to_csv(test_data_path, index=False)
    raw_training_data = raw_data[raw_data['Challenge Stats Submitby Date Date']<time_thr].reset_index(drop=True)
    print('training row number: '+str(len(raw_training_data)))
    raw_training_data.to_csv(training_data_path, index=False)