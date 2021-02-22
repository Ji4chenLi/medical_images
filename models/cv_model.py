# Note: Here we freeze the feature extractor, and only tune the MLC.
# We used the default learning rate of Adam to train the MLC.

# https://arxiv.org/pdf/2004.12274.pdf finetunes the feature extractor
# on the ChestX-ray 14 dataset.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class CNNnetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.dense_net_121 = tvm.densenet121(pretrained=True)
        num_fc_kernels = self.dense_net_121.classifier.in_features
        self.dense_net_121.classifier = nn.Linear(
            num_fc_kernels, num_fc_kernels
        )
        self._load_from_state_dict()
        self.dense_net_121.classifier.weight.data.copy_(
            torch.eye(num_fc_kernels)
        )
        self.dense_net_121.classifier.bias.data.zero_()

    def _load_from_state_dict(self, ckpt='./model.pth.tar'):
        pretrained_weight = torch.load(ckpt)['state_dict']
        new_state_dict = {}
        prefix = 'module.dense_net_121.'
        for k, v in pretrained_weight.items():
            if 'classifier' not in k:
                new_k = k[len(prefix):]
                new_state_dict[new_k] = v

        msg = self.dense_net_121.load_state_dict(new_state_dict, strict=False)
        assert set(msg.missing_keys) == {
            "classifier.weight",
            "classifier.bias"
        }, set(msg.missing_keys)

    def forward(self, x):
        with torch.no_grad():
            features = self.dense_net_121(x)

        return features

class MLC(nn.Module):
    def __init__(
        self,
        num_classes=13,
        fc_in_features=1024,
    ):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(
            in_features=fc_in_features,
            out_features=num_classes
        )

    def forward(self, visual_feature):
        return self.classifier(visual_feature)

    def get_tag_probs(self, visual_feature):
        tag_scores = self.forward(visual_feature)
        tag_probs = torch.sigmoid(tag_scores)
        return tag_probs

class MLCTrainer(nn.Module):
    def __init__(self, num_classes=13, threshold=0.5):
        super(MLCTrainer, self).__init__()
        self.mlc = MLC(num_classes)
        self.loss = nn.BCEWithLogitsLoss()
        self.threshold = threshold

    def forward(self, batch):
        tag_scores = self.mlc(batch.visual_feature)
        loss = self.loss(tag_scores, batch.label)

        probs = torch.sigmoid(tag_scores)
        preds = (probs > self.threshold).to(torch.float)
        return {"loss": loss, "preds": preds, "probs": probs}


## Training with the softmax crossentropy loss

# class MLCTrainer(nn.Module):
#     def __init__(self, threshold=0.5):
#         super(MLCTrainer, self).__init__()
#         self.mlc = MLC()
#         self.log_softmax = nn.LogSoftmax(dim=1)
#         self.threshold = threshold

#     def forward(self, batch):
#         tag_scores = self.mlc(batch.visual_feature)
#         loss = torch.mean(
#             -self.log_softmax(tag_scores) * batch.label
#         )

#         probs = torch.sigmoid(tag_scores)
#         preds = (probs > self.threshold).to(torch.float)
#         return {"loss": loss, "preds": preds, "probs": probs}


if __name__ == "__main__":
    m = CNNnetwork()
    print(m)
