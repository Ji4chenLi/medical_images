import pickle
import os.path as osp
import torch
import torch.utils.data

import texar as tx
from texar.torch.hyperparams import HParams
from texar.torch.data.data import DatasetBase
from texar.torch.data.data import DataSource
from texar.torch.data import Vocab

from forte.data.data_pack import DataPack
import config_findings as config


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

class IU_XRay_DataSource(DataSource):
    """
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())
        self.feature_root = self._hparams.feature_root
        self.file_names, self.labels = self.__load_label_list(
            self._hparams.label_root)
        self.text_root = self._hparams.text_root
        self.vocab = Vocab(self._hparams.vocab_path)

    def __load_label_list(self, file_list):
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]

                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __len__(self):
        return len(self.file_names)

    def __iter__(self):
        for i in range(len(self.file_names)):
            yield self.__getitem__(i)

    def __getitem__(self, index):
        index = int(index)

        label = torch.Tensor(self.labels[index])
        key = self.file_names[index]

        feature_path = osp.join(self.feature_root, key + '.pt')
        visual_feature = torch.load(feature_path)

        text_path = osp.join(self.text_root, key + '.json')

        with open(text_path, 'r') as f:
            items = list(DataPack.deserialize(f.read()))

        caption = items[1].content
        if caption is None:
            caption = 'the lungs are normal. '

        caption_token = list()
        max_word_num = 0

        for sentence in caption.split('. '):
            sentence = sentence.replace('.', '').split()
            if len(sentence) == 0 or len(sentence) == 1:
                continue
            tokens = self.vocab.map_tokens_to_ids_py(sentence).tolist()
            tokens.append(self.vocab.eos_token_id)

            max_word_num = max(max_word_num, len(tokens))
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
            "feature_root": None,
            "label_root": None,
            "text_root": None,
            "vocab_path": None,
            "batch_size": 32,
            "lazy_strategy": 'all',
            "cache_strategy": 'none',
            "shuffle": True,
            "shuffle_buffer_size": 32,
            "input_channel": "RGB",
            "num_parallel_calls": 1
        })
        return hparams


class IU_XRay_Dataset(DatasetBase):
    def __init__(self, hparams=None, device="cuda:0"):
        self.source = IU_XRay_DataSource(hparams)
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
            "feature_root": None,
            "label_root": None,
            "text_root": None,
            "vocab_path": None,
            "batch_size": 32,
            "lazy_strategy": 'all',
            "cache_strategy": 'none',
            "shuffle": True,
            "shuffle_buffer_size": 32,
            "input_channel": "RGB",
            "num_parallel_calls": 1
        })
        return hparams


if __name__ == "__main__":
    # hparams = config.dataset
    # dataset = IU_XRay_DataSource(hparams['train'])
    # # Dataloader
    # train_loader = torch.utils.data.DataLoader(dataset,
    #                                            batch_size=64,
    #                                            shuffle=True,
    #                                            num_workers=1,
    #                                            pin_memory=True,
    #                                            drop_last=True,
    #                                            collate_fn=collate_fn)

    # for batch in train_loader:
    #     print(batch)
    #     exit()

    import os
    count = 0
    root = './text_root'
    for i, filename in enumerate(os.listdir(root)):
        with open(osp.join(root, filename), 'r') as f:
            items = list(DataPack.deserialize(f.read()))

        caption = items[1].content
        if caption is not None:
            if 'active disease' in caption:
                print(filename)
                print(caption)
        else:
            print(items[2].content)

    print(count, i)
