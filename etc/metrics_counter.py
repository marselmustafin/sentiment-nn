import pandas as pd
import argparse
from sklearn.metrics import classification_report

GOLD_PATH = "scores/_scripts/SemEval2017_task4_subtaskA_test_english_gold.txt"

df = pd.read_csv(GOLD_PATH, sep="\t", names=['doc_id', 'sentiment'])

parser = argparse.ArgumentParser(description='metrics')
parser.add_argument('filepath', type=str, help='result file path')
args = parser.parse_args()

path = args.filepath.strip()

df2 = pd.read_csv(path, sep="\t", names=['doc_id', 'sentiment'])

gold_values = df.sentiment.values
model_values = df2.sentiment.values

print(classification_report(gold_values, model_values, digits=3))

