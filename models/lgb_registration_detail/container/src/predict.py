from config_and_helper import *
import argparse

def predict(model, input, output_path):
    print("Preprocessing data...")
    test_data_preprocessing(input).to_csv('cache/preprocessed_test_data.csv', index=False)
    test_data = pd.read_csv('cache/preprocessed_test_data.csv')
    test_labels = pd.read_csv('cache/test_labels.csv')
    column_list = list(test_data.columns)
    for i in range(len(column_list)):
        column_list[i] = column_list[i].replace('[','(').replace(']',')').replace(':','_').replace(',',' ')
    test_data.columns = column_list

    print("Start to predict...")
    output = pd.DataFrame()
    output['Success'] = model.predict(test_data)
    output['Challenge Stats Challenge ID'] = test_labels['Challenge Stats Challenge ID']
    output.to_csv(output_path, index=False)
    print("Ok!")

    print("F1 score: "+str(calculate_f1(test_labels, output)))
    test_labels.to_csv('cache/test_labels.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model predicting")
    parser.add_argument("model", help="model file", default='lgb')
    parser.add_argument("input", help="input test data file", default='raw_test_data.csv')
    parser.add_argument("--output", help="outpput file", default='predictions.csv', required=False)
    args = parser.parse_args()
    predict(joblib.load(args.model), pd.read_csv(args.input), args.output)
