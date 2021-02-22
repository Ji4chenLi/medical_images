import os.path as osp
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from config_findings import dataset as hparams_dataset


class preprocess_mimic():
    def __init__(self):
        self.pathologies = sorted(hparams_dataset["pathologies"])

        self.imgpath = hparams_dataset["imgpath"]
        self.csvpath = hparams_dataset["csvpath"]
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = hparams_dataset["metacsvpath"]
        self.metacsv = pd.read_csv(self.metacsvpath)
        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(self.metacsv).reset_index()

        self.mode = hparams_dataset["mode"]
        self.views = hparams_dataset["views"]
        self.prepare_csv_entries()
        self.save_csv()

    def prepare_csv_entries(self):
        healthy = self.csv["No Finding"] == 1
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[self.csv[pathology] == 1, pathology] = 1.0
                self.csv.loc[healthy, pathology] = 0.0
                self.csv.loc[self.csv[pathology] == -1, pathology] = 0.0
                self.csv.loc[pd.isna(self.csv[pathology]), pathology] = 0.0

        self.csv["path"] = self.csv.apply(
            lambda row: combine_path(self.imgpath, row), axis=1)

        if self.mode == 'PER_IMAGE':
            # Keep only the PA view.
            idx_pa = self.csv["ViewPosition"].isin(self.views)
            self.csv = self.csv[idx_pa]
            new_csv_column = ['path']
            for col_i in self.pathologies:
                new_csv_column.append(col_i)
            print(new_csv_column)
            self.csv = self.csv.filter(new_csv_column, axis=1)
        else:
            # grouping by study id
            self.csv['study'] = self.csv.apply(lambda x: str(Path(x['path']).parent), axis=1)
            self.csv.set_index(['study'], inplace=True)
            path_column_idx = self.csv.columns.get_loc('path')
            aggs = {self.csv.columns[path_column_idx]: lambda x: ','.join(x.astype(str))}
            aggs.update({x: 'mean' for x in self.pathologies})
            self.csv = self.csv.groupby(['study']).agg(aggs).reset_index(0, drop=True)

    def save_csv(self):
        dataset_indices = list(range(len(self.csv)))
        self.train_indices, test_indices = train_test_split(dataset_indices, test_size=0.3)
        self.val_indices, self.test_indices = train_test_split(test_indices, test_size=0.66)
        train_csv = self.csv.iloc[self.train_indices]
        train_csv.to_csv('mimic_train.csv', header=True, index=False)
        val_csv = self.csv.iloc[self.val_indices]
        val_csv.to_csv('mimic_val.csv', header=True, index=False)
        test_csv = self.csv.iloc[self.test_indices]
        test_csv.to_csv('mimic_test.csv', header=True, index=False)

def combine_path(imgpath, row):
    subjectid = str(row["subject_id"])
    studyid = str(row["study_id"])
    dicom_id = str(row["dicom_id"])
    img_path = osp.join(imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")

    return img_path


if __name__ == "__main__":
    # preprocess_mimic()
    # csv = pd.read_csv('/data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv')
    # for _, row in csv.iterrows():
    #     print(row)
    #     exit()
    csv = pd.read_csv('preprocessed/mimic_train.csv')
    count = [0] * 13
    for total, row in csv.iterrows():
        labels = row[1:].to_list()
        count[int(sum(labels))] += 1
    # count = [6969, 6843, 1281, 2152, 1066, 1385, 2159, 9658, 8373, 707, 4008, 1630, 3659]
    # total = 67307
    print(count, total)
    # print([round((total - c) / c, 1) for c in count])
 
 

## Codes to make sure the preprocessing is correct by adding the "No Finding" tags
    # import numpy as np
    # with_normal = pd.read_csv('preprocessed_with_normal/mimic_train.csv')
    # without_normal = pd.read_csv('without_normal_result/preprocessed/mimic_train.csv')

    # for i, with_normal_row in with_normal.iterrows():
    #     loc = (without_normal['path'] == with_normal_row[0]).to_numpy()
    #     try:
    #         index = np.where(loc)[0][0]
    #     except Exception:
    #         # print('1')
    #         continue

    #     without_normal_row = without_normal.loc[index]
    #     assert with_normal_row[0] == without_normal_row[0], (with_normal_row[0], without_normal_row[0])
    #     with_normal_label = with_normal_row[1:].tolist()
    #     without_normal_label = without_normal_row[1:].tolist()
        
    #     assert without_normal_label == with_normal_label
    #     print('Correct')

