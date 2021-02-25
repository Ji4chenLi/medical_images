import sys
import argparse
import os
import os.path as osp
from pathlib import Path
import torch
from texar.torch.run import Executor, cond, action
from texar.torch.run.metric import Average, RunningAverage

from models.model import MedicalReportGenerator
from iu_xray_data import IU_XRay_Dataset

from config_findings import dataset as hparams_dataset


# args
parser = argparse.ArgumentParser(description="Train IU XRay model")
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
    '--tbx_dir',
    type=str,
    help='Place to save tensorboard data',
    default='tbx_folder/'
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
    '--max_train_epochs',
    type=int,
    help='Maximum number of epochs to train',
    default=1000
)
args = parser.parse_args()

if not osp.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Dataloader
datasets = {split: IU_XRay_Dataset(hparams=hparams_dataset[split])
            for split in ["train", "val", "test"]}
print("done with loading")
# model
model = MedicalReportGenerator(hparams_dataset["model"])
output_dir = Path(args.output_dir)
optim = torch.optim.Adam(model.parameters(), lr=hparams_dataset["lr"])

# Trainer
executor = Executor(
    model=model,
    train_data=datasets["train"],
    valid_data=datasets["val"],
    test_data=datasets["test"],
    checkpoint_dir=args.save_dir,
    save_every=[cond.validation(better=True), cond.epoch(25)],
    train_metrics=[
        ("loss", Average(pred_name="loss")),
        ("stop_loss", Average(pred_name="stop_loss")),
        ("word_loss", Average(pred_name="word_loss")),
        ("attention_loss", Average(pred_name="attention_loss")),
        ("topic_var", RunningAverage(
            pred_name="topic_var", queue_size=args.display_steps)),
        ],
    optimizer=optim,
    grad_clip=args.grad_clip,
    log_every=cond.iteration(args.display_steps),
    log_destination=[sys.stdout, output_dir / "log.txt"],
    validate_every=cond.epoch(1),
    valid_metrics=[
        ("bleu_1", Average(pred_name="bleu_1")),
        ("bleu_2", Average(pred_name="bleu_2")),
        ("bleu_3", Average(pred_name="bleu_3")),
        ("bleu_4", Average(pred_name="bleu_4")),
        ("bleu", Average(pred_name="bleu")),
    ],
    plateau_condition=[
        cond.consecutive(cond.validation(better=False), 2)],
    action_on_plateau=[
        action.early_stop(patience=10),
        action.reset_params(),
        action.scale_lr(0.8)],
    stop_training_on=cond.epoch(args.max_train_epochs),
    test_mode='predict',
    tbx_logging_dir=args.tbx_dir,
    test_metrics=[
        ("bleu_1", Average(pred_name="bleu_1")),
        ("bleu_2", Average(pred_name="bleu_2")),
        ("bleu_3", Average(pred_name="bleu_3")),
        ("bleu_4", Average(pred_name="bleu_4")),
        ("bleu", Average(pred_name="bleu")),
    ],
    print_model_arch=False,
    show_live_progress=True
)

executor.train()
executor.test()
