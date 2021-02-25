import os
import os.path as osp
import csv
from tqdm import tqdm

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

class IU_XRay_Image_DataSource(DataSource):
    """
    Dataset website here:
    https://www.kaggle.com/raddar/chest-xrays-indiana-university
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())
        self.file_names, self.labels = self.__load_label_list(
            self._hparams.label_path)
        self.transforms = self.build_transform(self._hparams['transforms'])

    def __load_label_list(self, file_list):
        labels = []
        filename_list = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]

                image_name = '{}.png'.format(image_name)
                image_name = osp.join(self._hparams.img_root, image_name)

                filename_list.append(image_name)
                labels.append(label)
        return filename_list, labels

    def __len__(self):
        return len(self.file_names)

    def __iter__(self):
        for i in range(len(self.file_names)):
            yield self.__getitem__(i)

    def __getitem__(self, index):

        img_roots = self.file_names[index]
        image_tensor = self.get_image(img_roots, self.transforms)
        target_tensor = torch.FloatTensor(self.labels[index])

        return img_roots, image_tensor, target_tensor

    @staticmethod
    def build_transform(tsfm_list):
        t = []
        for func, args in tsfm_list:
            t.append(getattr(tfms, func)(**args))
        return tfms.Compose(t)

    def get_image(self, img_root, transforms):

        # In this way, we can skip the ToPILImage in the data augmentations,
        # speeding up the data loading
        image = pil_loader(img_root)
        image_tensor = transforms(image)
        return image_tensor

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.
        See the specific subclasses for the details.
        """
        hparams = DatasetBase.default_hparams()
        hparams.update({
            "img_root": None,
            "label_path": None,
            "transforms": None,
        })
        return hparams


class IU_XRay_Image_Dataset(DatasetBase):
    def __init__(self, hparams=None, device="cuda:0"):
        self.source = IU_XRay_Image_DataSource(hparams)
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
            "img_root": None,
            "transforms": None,
            "label_path": None
        })
        return hparams


def save_features_pt(loader, root_dir):
    with torch.no_grad():
        for batch in tqdm(loader):
            paths = batch['path']
            features = model(batch['image'])
            for path, feature in zip(paths, features):
                path = path.replace(prefix, '').replace('.png', '.pt')
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
    model = CNNnetwork()
    prefix = "/home/jiachen.li/iu_xray_images/"
    root_to_save = '/home/jiachen.li/data_iu_xray'
    transforms = [
        ("Resize", {
            "size": 256,
            "interpolation": 1
        }),
        ("CenterCrop", {
            "size": 224
        }),
        ("ToTensor", {}),
        ("Normalize", {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        })
    ]
    # Train
    hparams = {
        "img_root": prefix,
        "transforms": transforms,
        "label_path": './mlc_data/train_data.txt' 
    }
    dataset_train = IU_XRay_Image_Dataset(hparams=hparams)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=32,
                                               num_workers=2,
                                               drop_last=False)

    for i, batch in tqdm(enumerate(train_loader)):
        print(i)
