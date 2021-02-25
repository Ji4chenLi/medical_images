import xml.etree.ElementTree as ET
import json
import torch
from texar.torch.data.data.data_iterators import DataIterator
from forte.data.readers.deserialize_reader import RawDataDeserializeReader
from config_findings import dataset as hparams_dataset
from iu_xray_data import IU_XRay_Dataset
from models.model import MedicalReportGenerator


class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)
        self.keys = list(self.data.keys())

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        return self.data[item]
        # return self.data[self.keys[item]]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    m = MedicalReportGenerator(hparams_dataset["model"])
    # params = torch.load("exp_default_lstm_full/1614127379.7966576.pt")
    # params = torch.load("exp_default_lstm/1614031943.4646807.pt")
    params = torch.load("exp_default_lstm_epoch_1000/1614185928.6635778.pt")
    m.load_state_dict(params.model)
    datasets = {split: IU_XRay_Dataset(hparams=hparams_dataset[split])
            for split in ["train", "val", "test"]}
    datasets["train"].to(torch.device('cuda'))

    iterator = DataIterator(datasets["train"])

    for i, batch in enumerate(iterator):
        r = m.predict(batch)
        # r = m(batch)
        if i > 2:
            exit()

    # from texar.torch.evals import corpus_bleu_moses, corpus_bleu
    # candidate_corpus = [
    #     ['My', 'full', 'pytorch', 'test'],
    #     ['Another', 'Sentence']
    # ]
    # references_corpus = [
    #     [
    #         ['My', 'full', 'pytorch', 'test'],
    #         ['Completely', 'Different']
    #     ],
    #     [
    #         ['No', 'Match']
    #     ]
    # ]
    # print(corpus_bleu_moses(list_of_references=references_corpus, hypotheses=candidate_corpus, return_all=True))
    #######################################################################################

    # tree = ET.parse("./333.xml")
    # root = tree.getroot()

    # abs_text_list = []
    # for abs_text in list(root.find('MedlineCitation/Article/Abstract')):
    #     if abs_text.attrib['Label'] in ['FINDINGS', 'IMPRESSION']:
    #         text = abs_text.text if abs_text.text else ' '
    #         content = abs_text.attrib['Label'] + ': ' + abs_text.text
    #         abs_text_list.append(content)

    # img_name_list = []
    # for node in list(root):
    #     if node.tag == 'parentImage':
    #         img_name_list.append('PARENT IMAGE: ' + node.find('./panel/url').text)
    # text = ', '.join(abs_text_list + img_name_list)
    # print(text)

    #######################################################################################

    # labels = []
    # filename_list = []
    # with open('./mlc_data/test_data.txt', 'r') as f:
    #     for line in f:
    #         items = line.split()
    #         image_name = items[0]
    #         label = items[1:]
    #         label = [int(i) for i in label]
    #         image_name = '{}.png'.format(image_name)
    #         filename_list.append(image_name)
    #         labels.append(label)
    # print(len(labels))

    #######################################################################################

    # reader = RawDataDeserializeReader()
    # for item in reader._parse_pack('./result/CXR1_1_IM-0001-3001.json'):
    #     print(item)
