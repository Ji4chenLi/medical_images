import torch
from network import ClassifierWrapper

from mimic_dataset import MIMICCXR_Dataset
from config_mimic_test import dataset as hparams_dataset


def train():
    extractor = ClassifierWrapper()


if __name__ == "__main__":
    train()
