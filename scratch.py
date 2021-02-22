import torch
from texar.torch.data.data.data_iterators import DataIterator

from config_findings import dataset as hparams_dataset
from mimic_cxr import MIMICCXR_Dataset
from models.model import MedicalReportGenerator
from build_vocab import Vocabulary


if __name__ == "__main__":

    m = MedicalReportGenerator(hparams_dataset["model"])
    params = torch.load("exp_default_lstm_100/1613636897.723331.pt")
    m.load_state_dict(params.model)
    datasets = {split: MIMICCXR_Dataset(hparams=hparams_dataset[split])
            for split in ["train", "val", "test"]}
    datasets["train"].to(torch.device('cuda'))

    iterator = DataIterator(datasets["train"])

    for batch in iterator:
        # r = m.predict(batch)
        r = m(batch)
        exit()
