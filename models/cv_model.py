# Note: Here we freeze the feature extractor, and only tune the MLC.
# We used the default learning rate of Adam to train the MLC.

# https://arxiv.org/pdf/2004.12274.pdf finetunes the feature extractor
# on the ChestX-ray 14 dataset.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class CNNnetwork(nn.Module):

    def __init__(self, mode="PER_IMAGE", input_channel="RGB"):
        super().__init__()
        self.mode = mode
        assert input_channel == "RGB"
        densenet = tvm.densenet121(pretrained=True)
        modules = list(densenet.features)
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        assert self.mode == "PER_IMAGE"
        with torch.no_grad():
            z = self.features(x)
            z = F.relu(z)
            features = F.adaptive_avg_pool2d(z, (1, 1)).squeeze()

        return features

class MLC(nn.Module):
    def __init__(
        self,
        num_classes=14,
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
    def __init__(self, threshold=0.5):
        super(MLCTrainer, self).__init__()
        self.mlc = MLC()
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
    m.to_distributed("cuda:0")
