from config_and_helper import *
import argparse
import os

def train(data, model_path):
    print("Preprocessing data...")
    raw_data = pd.read_csv(data).sample(frac=1, random_state=32, replace=False)
    data_output, label_output, _, _, extended_columns = training_data_preprocessing(raw_data)
    data_output.to_csv('cache/preprocessed_training_data.csv', index=False)
    label_output.to_csv('cache/preprocessed_training_labels.csv', index=False)
    with open("cache/extended_columns.pkl", "wb") as f:
        pickle.dump(extended_columns, f)
    all_training_data = pd.read_csv('cache/preprocessed_training_data.csv').sample(frac=1, random_state=25).reset_index(drop=True)
    all_training_labels = pd.read_csv('cache/preprocessed_training_labels.csv').sample(frac=1, random_state=25).reset_index(drop=True)

    print("Start to train...")
    params = {'bagging_fraction': 0.78, 'boosting': 'goss', 'feature_fraction': 0.73, 'is_unbalance': True, 
              'lambda_l1': 0.9167131900525011, 'lambda_l2': 3.752785416817357, 'learning_rate': 0.0702668621274247, 
              'max_bin': 106, 'max_depth': 12, 'min_data_in_bin': 96, 'min_data_in_leaf': 18, 'min_gain_to_split': 2.6,
              'num_leaves': 1262, 'objective': 'binary', 'other_rate': 0.39930653486883316, 'top_rate': 0.14712220010474805}
    column_list = list(all_training_data.columns)
    for i in range(len(column_list)):
        column_list[i] = column_list[i].replace('[','(').replace(']',')').replace(':','_').replace(',',' ')
    all_training_data.columns = column_list
    training_data = all_training_data
    training_labels = all_training_labels
    test_data = all_training_data[len(all_training_data)*5//6:]
    test_labels = all_training_labels[len(all_training_labels)*5//6:]
    feature_importance, final_score = train_model(training_data, training_labels, test_data, test_labels, params, 
                                                  plot_feature_importance=True, model_path_name=model_path)
    feature_importance.to_csv('other_resources/feature_importance.csv')
    print("Ok! output mode: "+model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("data", help="input data path", default='raw_training_data.csv')
    parser.add_argument("model", help="outpput model path", default='lgb')
    args = parser.parse_args()
    train(args.data, args.model)
