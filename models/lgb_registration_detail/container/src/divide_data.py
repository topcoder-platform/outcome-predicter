from config_and_helper import *
import argparse

parser = argparse.ArgumentParser(description="Model training")
parser.add_argument("raw_data", help="input raw data path", default='challenge-with-registration.xlsx')
parser.add_argument("raw_training_data", help="outpput raining data path", default='raw_training_data.csv')
parser.add_argument("raw_test_data", help="outpput test data path", default='raw_test_data.csv')
args = parser.parse_args()

raw_data = pd.read_excel(args.raw_data)
divide_data(raw_data, args.raw_training_data, args.raw_test_data)