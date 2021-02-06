import sys
import argparse
import os
import os.path as osp
from pathlib import Path
import torch
from texar.torch.run import Executor, cond, metric, action

from build_vocab import Vocabulary
from models.cv_model import MLCTrainer
from mimic_cxr import MIMICCXR_Dataset

from evaluation_metrics import HammingLoss, MultiLabelConfusionMatrix, \
    MultiLabelF1, MultiLabelPrecision, MultiLabelRecall, RocAuc

from config_findings import dataset as hparams_dataset


# args
parser = argparse.ArgumentParser(description="Train MIMIC model")
parser.add_argument(
    '--save_dir',
    type=str,
    help='Place to save training results',
    default='exp_default/'
)
parser.add_argument(
    '--output_dir',
    type=str,
    help='Place to save logs results',
    default='output_default/'
)
parser.add_argument(
    '--grad_clip',
    type=float,
    help='Gradient clip value',
    default=None
)
parser.add_argument(
    '--display_steps',
    type=int,
    help='log result every * steps',
    default=1
)
parser.add_argument(
    '--max_train_steps',
    type=int,
    help='Maximum number of steps to train',
    default=1000000
)
args = parser.parse_args()

if not osp.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Dataloader
datasets = {split: MIMICCXR_Dataset(hparams=hparams_dataset[split])
            for split in ["train", "val", "test"]}
print("done with loading")
# model
model = MLCTrainer()
output_dir = Path(args.output_dir)
num_label = len(hparams_dataset['pathologies'])

# Trainer
executor = Executor(
    model=model,
    train_data=datasets["train"],
    valid_data=datasets["val"],
    test_data=datasets["test"],
    checkpoint_dir=args.save_dir,
    save_every=cond.validation(better=True),
    train_metrics=[("loss", metric.RunningAverage(args.display_steps))],
    optimizer={"type": torch.optim.Adam},
    grad_clip=args.grad_clip,
    log_every=cond.iteration(args.display_steps),
    log_destination=[sys.stdout, output_dir / "log.txt"],
    validate_every=cond.epoch(1),
    valid_metrics=[
        # HammingLoss[float](num_label=num_label, pred_name="preds", label_name="label"),
        # RocAuc(pred_name="probs", label_name="label"),
        # MultiLabelConfusionMatrix(num_label=num_label, pred_name="preds", label_name="label"),
        # MultiLabelPrecision(num_label=num_label, pred_name="preds", label_name="label"),
        # MultiLabelRecall(num_label=num_label, pred_name="preds", label_name="label"),
        # MultiLabelF1(num_label=num_label, pred_name="preds", label_name="label"),
        ("loss", metric.Average())
    ],
    plateau_condition=[
        cond.consecutive(cond.validation(better=False), 2)],
    action_on_plateau=[
        action.early_stop(patience=10),
        action.reset_params(),
        action.scale_lr(0.8)],
    stop_training_on=cond.iteration(args.max_train_steps),
    test_mode='eval',
    tbx_logging_dir='tbx_folder',
    test_metrics=[
        HammingLoss[float](num_label=num_label, pred_name="preds", label_name="label"),
        RocAuc(pred_name="probs", label_name="label"),
        MultiLabelConfusionMatrix(num_label=num_label, pred_name="preds", label_name="label"),
        MultiLabelPrecision(num_label=num_label, pred_name="preds", label_name="label"),
        MultiLabelRecall(num_label=num_label, pred_name="preds", label_name="label"),
        MultiLabelF1(num_label=num_label, pred_name="preds", label_name="label"),
        ("loss", metric.Average())
    ],
    print_model_arch=False,
    show_live_progress=True
)

# executor.train()
executor.load('exp_default/1611883760.6407511.pt')
executor.test()
