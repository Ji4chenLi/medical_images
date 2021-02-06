import pickle
import csv
import os.path as osp
import pandas as pd
from utils import load_txt, is_nan
from config_findings import dataset


text_data_path = './preprocessed/text_data.pkl'


def clean_txt(phrases):
    if not is_nan(phrases):
        phrases = phrases.replace('- ', '').replace('/', '')
    return phrases

def process_txt():
    text_dict = {}
    prefix = dataset['imgpath']
    text = pd.read_csv('./mimic_text_full.csv')
    for _, item in text.iterrows():
        key = osp.join(prefix, item['path'][:-4])
        findings = clean_txt(item['findings'])
        impression = clean_txt(item['impression'])

        text_dict[key] = {
            'findings': findings,
            'impression': impression
        }

    with open('./preprocessed/text_data.pkl', 'wb') as f:
        pickle.dump(text_dict, f)



def check_item_exist():
    text_dict = load_txt(text_data_path)
    image = pd.read_csv('./preprocessed_impression/mimic_train.csv')
    for _, item in image.iterrows():
        paths = item[0].split(',')
        key = '/'.join(paths[0].split('/')[:-1])
        findings = text_dict[key]['findings']
        impression = text_dict[key]['impression']

        if not is_nan(findings):
            if '-' in findings or '/' in findings:
                print(findings)

def preprocessed_findings():
    text_dict = load_txt(text_data_path)
    inp_root = './preprocessed'
    out_root = './preprocessed_impressions'
    sub_files = ['mimic_train.csv', 'mimic_val.csv', 'mimic_test.csv']
    for sf in sub_files:
        inp_file = osp.join(inp_root, sf)
        out_file = osp.join(out_root, sf)
        with open(inp_file, 'rt') as inp, open(out_file, 'wt') as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                paths = row[0].split(',')
                key = '/'.join(paths[0].split('/')[:-1])
                findings = text_dict[key]['findings']
                if findings == findings:
                    writer.writerow(row)

def preprocessed_impression():
    text_dict = load_txt(text_data_path)
    inp_root = './preprocessed'
    out_root = './preprocessed_impression'
    sub_files = ['mimic_train.csv', 'mimic_val.csv', 'mimic_test.csv']
    for sf in sub_files:
        inp_file = osp.join(inp_root, sf)
        out_file = osp.join(out_root, sf)
        with open(inp_file, 'rt') as inp, open(out_file, 'wt') as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                paths = row[0].split(',')
                key = '/'.join(paths[0].split('/')[:-1])
                impression = text_dict[key]['impression']
                if impression == impression:
                    writer.writerow(row)


if __name__ == "__main__":
    process_txt()
    # load_txt(text_data_path)
    # preprocessed_findings()
    # preprocessed_impression()
    # check_item_exist()
