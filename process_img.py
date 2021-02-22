import os
import os.path as osp
import csv
from tqdm import tqdm
import pandas as pd

import torch
import torch.utils.data
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as tfms

import texar as tx
from texar.torch.hyperparams import HParams
from texar.torch.data.data import DatasetBase
from texar.torch.data.data import DataSource

import config_findings as config
from models.cv_model import CNNnetwork

class MIMICCXR_Image_DataSource(DataSource):
    """
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())
        self.mode = self._hparams["mode"]
        self.csv = pd.read_csv(self._hparams["processed_csv"])
        self.transforms = self.build_transform(self._hparams['transforms'])

    def __len__(self):
        return len(self.csv)

    def __iter__(self):
        for _, row in self.csv.iterrows():
            yield row

    def __getitem__(self, index):
        index = int(index)

        def get_entries(index):
            df = self.csv.iloc[index]
            paths = df[0].split(',')
            label = df[1:].tolist()
            return paths, label

        assert self.mode == "PER_IMAGE"
        img_paths, label = get_entries(index)
        image_tensor = self.get_image(img_paths[0], self.transforms)
        target_tensor = torch.FloatTensor(label)
        return img_paths, image_tensor, target_tensor

    @staticmethod
    def build_transform(tsfm_list):
        t = []
        for func, args in tsfm_list:
            t.append(getattr(tfms, func)(**args))
        return tfms.Compose(t)

    def get_image(self, img_path, transforms):
        assert self._hparams["input_channel"] != "GRAY"

        # In this way, we can skip the ToPILImage in the data augmentations,
        # speeding up the data loading
        image = pil_loader(img_path)
        image_tensor = transforms(image)
        return image_tensor

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


class MIMICCXR_Image_Dataset(DatasetBase):
    def __init__(self, hparams=None, device="cuda:0"):
        self.source = MIMICCXR_Image_DataSource(hparams)
        super().__init__(self.source, hparams, device)

    def process(self, raw_example):
        return {
            "path": raw_example[0],
            "image": raw_example[1],
            "target": raw_example[2]
        }

    def collate(self, examples):
        # `examples` is a list of objects returned from the
        # `process` method. These data examples should be collated
        # into a batch.

        # `images` is a `tensor` of input images, storing the transformed
        # images for each example in the batch.

        # `target` is the one hot encoding of the labels with the size of
        # number of classes, stack into the batch

        images = []
        paths = []
        targets = []

        for ex in examples:
            paths.append(ex["path"])
            images.append(ex["image"])
            targets.append(ex["target"])

        images = torch.stack(images)
        targets = torch.stack(targets)

        return tx.torch.data.Batch(
            len(examples),
            image=images,
            target=targets,
            path=paths)

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
            "transforms": None,
            "mode": None,
            "batch_size": 1,
            "shuffle": False,
            "shuffle_buffer_size": 32,
            "input_channel": "RGB"
        })
        return hparams


def save_features_pt(loader, root_dir):
    with torch.no_grad():
        for batch in tqdm(loader):
            paths = batch['path'][0]
            features = model(batch['image'])
            for path, feature in zip(paths, features):
                path = path.replace(prefix, '').replace('.jpg', '.pt')
                path = path[1:]
                path = osp.join(root_dir, path)
                folder = osp.dirname(path)
                if not osp.exists(folder):
                    os.makedirs(folder)

                torch.save(feature, path)

def exclude_invalid(inp_file, out_file, invalid):
    with open(inp_file, 'rt') as inp, open(out_file, 'wt') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            paths = row[0].split(',')[0]
            if invalid[0] in paths or invalid[1] in paths or invalid[2] in paths:
                continue
            else:
                writer.writerow(row)


if __name__ == "__main__":
    hparams = config.dataset
    datasets = {split: MIMICCXR_Image_Dataset(hparams=config.dataset[split])
                for split in ["train", "val", "test"]}
    model = CNNnetwork()

    prefix = config.dataset['imgpath']

    train_loader = torch.utils.data.DataLoader(datasets['train'],
                                               batch_size=32,
                                               num_workers=2,
                                               drop_last=False)

    root_to_save = '/home/jiachen.li/data_finetune'
    save_features_pt(train_loader, root_to_save)

    val_loader = torch.utils.data.DataLoader(datasets['val'],
                                               batch_size=32,
                                               num_workers=2,
                                               drop_last=False)

    root_to_save = '/home/jiachen.li/data_finetune'
    save_features_pt(val_loader, root_to_save)

    test_loader = torch.utils.data.DataLoader(datasets['test'],
                                               batch_size=32,
                                               num_workers=2,
                                               drop_last=False)

    root_to_save = '/home/jiachen.li/data_finetune/test'
    save_features_pt(test_loader, root_to_save)


    # invalid = [
    #     '8de3cbff-0613dea5-597b3a9b-cf3bc5e6-f87f6c36.jpg',
    #     'bef65ae1-4e634fec-87c5648e-2310b295-352456b0.jpg',
    #     '2e75ef66-664a337d-927a64a7-c2c87db7-2f2688dc.jpg'
    # ]

    # inp_file = hparams['train']['processed_csv']
    # out_file = './preprocessed/train.csv'
    # exclude_invalid(inp_file, out_file, invalid)

    # inp_file = hparams['val']['processed_csv']
    # out_file = './preprocessed/val.csv'
    # exclude_invalid(inp_file, out_file, invalid)

    # inp_file = hparams['test']['processed_csv']
    # out_file = './preprocessed/test.csv'
    # exclude_invalid(inp_file, out_file, invalid)

######################

    # train_csv = pd.read_csv(hparams['train']['processed_csv'])
    # val_csv = pd.read_csv(hparams['val']['processed_csv'])
    # test_csv = pd.read_csv(hparams['test']['processed_csv'])

    # feature_root = '/home/jiachen.li/data_finetune'
    # prefix = hparams['imgpath']

    # count = 0
    # for _, row in train_csv.iterrows():
    #     path = row[0].split(',')[0]
    #     pt_path = osp.join(feature_root, path.replace(prefix, '').replace('.jpg', '.pt')[1:])

    #     if not osp.exists(pt_path):
    #         # img = pil_loader(path)
    #         # img_pt = datasets['train'].source.transforms(img).unsqueeze(0)
    #         # with torch.no_grad():
    #         #     print(img_pt.shape)
    #         #     feature = model(img_pt)
    #         # torch.save(feature, path)
    #         count += 1
    #         continue

    # print(count)

    # for _, row in val_csv.iterrows():
    #     path = row[0].split(',')[0]
    #     pt_path = osp.join(feature_root, path.replace(prefix, '').replace('.jpg', '.pt')[1:])

    #     if not osp.exists(pt_path):
    #         # img = pil_loader(path)
    #         # img_pt = datasets['val'].source.transforms(img).unsqueeze(0)
    #         # with torch.no_grad():
    #         #     print(img_pt.shape)
    #         #     feature = model(img_pt)
    #         # torch.save(feature, path)
    #         count += 1
    #         continue

    # print(count)

    # for _, row in test_csv.iterrows():
    #     path = row[0].split(',')[0]
    #     pt_path = osp.join(feature_root, path.replace(prefix, '').replace('.jpg', '.pt')[1:])

    #     if not osp.exists(pt_path):
    #         # img = pil_loader(path)
    #         # img_pt = datasets['test'].source.transforms(img).unsqueeze(0)
    #         # with torch.no_grad():
    #         #     print(img_pt.shape)
    #         #     feature = model(img_pt)
    #         # torch.save(feature, path)
    #         count += 1
    #         continue

    # print(count)
