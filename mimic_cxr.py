import pickle
import os.path as osp
import pandas as pd
import torch
import torch.utils.data

import texar as tx
from texar.torch.hyperparams import HParams
from texar.torch.data.data import DatasetBase
from texar.torch.data.data import DataSource
from build_vocab import Vocabulary
import config_findings as config
from utils import is_nan


def collate_fn(data):
    visual_feature, label, caption_token, \
        max_word_num, sentence_num = zip(*data)

    visual_feature = torch.stack(visual_feature, 0)
    label = torch.stack(label, 0)

    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)
    batch_size = len(caption_token)

    token_tensor = torch.zeros([batch_size, max_sentence_num + 1, max_word_num])
    stop_prob = torch.zeros([batch_size, max_sentence_num + 1])

    for i, token in enumerate(caption_token):
        for j, sentence in enumerate(token):
            token_tensor[i, j, :len(sentence)] = torch.Tensor(sentence)
            stop_prob[i][j] = len(sentence) > 0

    return visual_feature, label, token_tensor, stop_prob

class MIMICCXR_DataSource(DataSource):
    """
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())
        self.prefix = self._hparams['imgpath']
        self.feature_root = self._hparams['feature_root']
        self.mode = self._hparams["mode"]
        self.image_csv = pd.read_csv(self._hparams["processed_csv"])
        self.txt_dict = pickle.load(open(self._hparams["txtpath"], 'rb'))
        self.vocab = pickle.load(open(self._hparams["vocabpath"], 'rb'))

    def __len__(self):
        return len(self.image_csv)

    def __iter__(self):
        for _, row in self.image_csv.iterrows():
            yield row

    def __getitem__(self, index):
        index = int(index)

        def get_entries(index):
            df = self.image_csv.iloc[index]
            path = df[0].split(',')[0]
            key = '/'.join(path.split('/')[:-1])
            label = df[1:].tolist()

            path = path.replace(self.prefix, '').replace('.jpg', '.pt')
            path = path[1:]
            path = osp.join(self.feature_root, path)

            return path, label, key

        assert self.mode == "PER_IMAGE"
        feature_path, label, key = get_entries(index)
        visual_feature = torch.load(feature_path)

        label = torch.Tensor(label)

        caption = self.txt_dict[key]['findings']
        if is_nan(caption):
            caption = 'the lungs are normal. '
        caption_token = list()
        max_word_num = 0

        for sentence in caption.split('. '):
            sentence = sentence.split()
            if len(sentence) == 0 or len(sentence) == 1:
                continue

            tokens = [self.vocab(token) for token in sentence]
            tokens.append(self.vocab('<end>'))
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            caption_token.append(tokens)
        sentence_num = len(caption_token)

        return visual_feature, label, caption_token, max_word_num, sentence_num

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.
        See the specific subclasses for the details.
        """
        hparams = DatasetBase.default_hparams()
        hparams.update({
            "imgpath": None,
            "feature_root": None,
            "txtpath": None,
            "vocabpath": None,
            "transforms": None,
            "processed_csv": None,
            "mode": None,
            "batch_size": 1,
            "input_channel": "RGB"
        })
        return hparams


class MIMICCXR_Dataset(DatasetBase):
    def __init__(self, hparams=None, device="cuda:0"):
        self.source = MIMICCXR_DataSource(hparams)
        super().__init__(self.source, hparams, device)

    def collate(self, examples):
        visual_feature, label, token_tensor, stop_prob = collate_fn(examples)

        return tx.torch.data.Batch(
            len(examples),
            visual_feature=visual_feature,
            label=label,
            token_tensor=token_tensor,
            stop_prob=stop_prob,
        )

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.
        See the specific subclasses for the details.
        """
        hparams = DatasetBase.default_hparams()
        hparams.update({
            "imgpath": None,
            "processed_csv": None,
            "feature_root": None,
            "txtpath": None,
            "vocabpath": None,
            "mode": None,
            "batch_size": 1,
            "shuffle": False,
            "transforms": None,
            "shuffle_buffer_size": 32,
            "input_channel": "RGB"
        })
        return hparams


if __name__ == "__main__":
    hparams = config.dataset
    dataset = MIMICCXR_DataSource(hparams['train'])
    # Dataloader
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )

    # disable data aug
    valid_dataset.data_aug = None

    train_dataset.csv = dataset.image_csv.iloc[train_dataset.indices]
    valid_dataset.csv = dataset.image_csv.iloc[valid_dataset.indices]
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=collate_fn)

    for batch in train_loader:
        print(batch)
        break
