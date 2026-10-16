#!/usr/bin/env python3
"""Train the SpliSync ablation models reported in the manuscript.

* random_hyena: frozen randomly initialized eight-layer HyenaDNA + pile-up.
* alignment_only: zero sequence channels + pile-up.
* sequence_only: frozen pretrained HyenaDNA + zero alignment channels.
* onehot_cnn: trainable one-hot CNN + pile-up.

The implementation imports model and data definitions from the repository's
``run_splisync.py`` using repository-relative paths; no local absolute source
paths are required.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SPLISYNC_SCRIPT = REPO_ROOT / "run_splisync.py"
ABLATIONS = (
    "random_hyena",
    "alignment_only",
    "sequence_only",
    "onehot_cnn",
)


def _load_splisync_module():
    """Import run_splisync.py without invoking its CLI."""
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "_splisync_ablation_base",
        SPLISYNC_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import SpliSync definitions from {SPLISYNC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


splisync = _load_splisync_module()


def read_hyena_config(
    pretrained_ckpt_dir: str,
    pretrained_name: str,
    backbone_cfg: Optional[dict] = None,
) -> dict:
    if backbone_cfg is not None:
        config = copy.deepcopy(backbone_cfg)
    else:
        config_path = (
            Path(pretrained_ckpt_dir).expanduser().resolve()
            / pretrained_name
            / "config.json"
        )
        with config_path.open() as config_file:
            config = json.load(config_file)

    if config.get("n_layer") != 8:
        raise ValueError(
            "The manuscript ablations use the eight-layer HyenaDNA "
            f"configuration; received n_layer={config.get('n_layer')!r}."
        )
    return config


def pileup_features(pileup_embed: torch.Tensor) -> torch.Tensor:
    embeddings = splisync.LinearReLUEmbeddings(
        10000,
        2,
        128,
    )
    feature_encoder = nn.Sequential(
        embeddings,
        nn.Flatten(start_dim=2),
    )
    feature_encoder.to(pileup_embed.device)

    upsample_factor = torch.empty(
        (pileup_embed.shape[0], 1, 1),
        dtype=pileup_embed.dtype,
        device=pileup_embed.device,
    ).uniform_(1.0, 3.0)
    transformed = torch.log1p(
        torch.round(pileup_embed[:, :, :2] * upsample_factor)
    )
    return feature_encoder(transformed)


class AlignmentOnlyModel(nn.Module):
    """Pile-up branch with zero-filled sequence channels."""

    def __init__(self) -> None:
        super().__init__()
        self.head = splisync.UNet1D(in_channels=512)

    def forward(
        self,
        input_ids,
        pileup_embed,
        position_ids=None,
        state=None,
    ):
        del input_ids, position_ids, state
        alignment_features = pileup_features(pileup_embed)
        sequence_zeros = torch.zeros_like(alignment_features)
        return self.head(
            torch.cat((sequence_zeros, alignment_features), dim=2)
        )


class SequenceOnlyModel(splisync.HyenaDNAModel):
    """Frozen pretrained HyenaDNA with zero-filled alignment channels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head = splisync.UNet1D(in_channels=512)

    def forward(
        self,
        input_ids,
        pileup_embed,
        position_ids=None,
        state=None,
    ):
        del pileup_embed, state
        sequence_features = self.backbone(
            input_ids,
            position_ids=position_ids,
        )
        alignment_zeros = torch.zeros_like(sequence_features)
        return self.head(
            torch.cat((sequence_features, alignment_zeros), dim=2)
        )


class OneHotCNNEncoder(nn.Module):
    """Small task-specific one-hot CNN producing 256 channels per base."""

    def __init__(self) -> None:
        super().__init__()
        # CharacterTokenizer IDs: A=7, C=8, G=9, T=10, N/PAD=4.
        token_to_channel = torch.full((12,), 4, dtype=torch.long)
        token_to_channel[7] = 0
        token_to_channel[8] = 1
        token_to_channel[9] = 2
        token_to_channel[10] = 3
        self.register_buffer("token_to_channel", token_to_channel)

        self.network = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.GELU(),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.min() < 0 or input_ids.max() >= len(self.token_to_channel):
            raise ValueError("Input contains a token outside the configured vocabulary")
        channels = self.token_to_channel[input_ids.long()]
        one_hot = F.one_hot(channels, num_classes=5)
        one_hot = one_hot.to(dtype=torch.float32).permute(0, 2, 1)
        return self.network(one_hot).permute(0, 2, 1)


class OneHotCNNModel(nn.Module):
    """One-hot CNN sequence branch combined with the pile-up branch."""

    def __init__(self) -> None:
        super().__init__()
        self.sequence_encoder = OneHotCNNEncoder()
        self.head = splisync.UNet1D(in_channels=512)

    def forward(
        self,
        input_ids,
        pileup_embed,
        position_ids=None,
        state=None,
    ):
        del position_ids, state
        sequence_features = self.sequence_encoder(input_ids)
        alignment_features = pileup_features(pileup_embed)
        return self.head(
            torch.cat((sequence_features, alignment_features), dim=2)
        )


def load_pretrained_sequence_model(
    pretrained_ckpt_dir: str,
    pretrained_name: str,
    backbone_cfg: Optional[dict] = None,
) -> SequenceOnlyModel:
    """Build the sequence-only model and load the pretrained backbone."""
    config = read_hyena_config(
        pretrained_ckpt_dir,
        pretrained_name,
        backbone_cfg,
    )
    model = SequenceOnlyModel(
        **copy.deepcopy(config),
        use_head=False,
        n_classes=2,
        stage="train",
    )

    checkpoint_path = (
        Path(pretrained_ckpt_dir).expanduser().resolve()
        / pretrained_name
        / "weights.ckpt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pretrained_state = checkpoint.get("state_dict")
    if pretrained_state is None:
        raise KeyError(f"Checkpoint has no 'state_dict': {checkpoint_path}")
    state_dict = splisync.load_weights(
        model.state_dict(),
        pretrained_state,
        checkpointing=bool(config.get("checkpoint_mixer", False)),
    )
    model.load_state_dict(state_dict)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


def build_random_hyena_model(
    pretrained_ckpt_dir: str,
    pretrained_name: str,
    backbone_cfg: Optional[dict] = None,
) -> splisync.HyenaDNAModel:
    """Build and freeze HyenaDNA without loading pretrained weights."""
    config = read_hyena_config(
        pretrained_ckpt_dir,
        pretrained_name,
        backbone_cfg,
    )
    model = splisync.HyenaDNAModel(
        **copy.deepcopy(config),
        use_head=False,
        n_classes=2,
        stage="train",
    )
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


class AblationLightningModule(pl.LightningModule):
    """Common optimization wrapper for all reported ablations."""

    def __init__(
        self,
        ablation: str,
        pretrained_ckpt_dir: str,
        pretrained_name: str = "hyenadna-medium-160k-seqlen",
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 100,
        backbone_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation {ablation!r}; choose from {ABLATIONS}")

        self.save_hyperparameters(
            {
                "ablation": ablation,
                "pretrained_ckpt_dir": pretrained_ckpt_dir,
                "pretrained_name": pretrained_name,
                "lr": lr,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
            }
        )

        if ablation == "random_hyena":
            self.model = build_random_hyena_model(
                pretrained_ckpt_dir,
                pretrained_name,
                backbone_cfg,
            )
        elif ablation == "alignment_only":
            self.model = AlignmentOnlyModel()
        elif ablation == "sequence_only":
            self.model = load_pretrained_sequence_model(
                pretrained_ckpt_dir,
                pretrained_name,
                backbone_cfg,
            )
        else:
            self.model = OneHotCNNModel()

        self.loss_fn = torchvision.ops.focal_loss.sigmoid_focal_loss
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        print(f"Ablation: {ablation}")
        print(f"Parameters: {total:,} total; {trainable:,} trainable")

    def forward(self, sequence, pileup_embed=None):
        return self.model(sequence, pileup_embed=pileup_embed)

    def training_step(self, batch, batch_idx):
        sequence, target, pileup = batch
        logits = self(sequence, pileup_embed=pileup)
        loss = self.loss_fn(
            logits.squeeze(),
            target.squeeze(),
            alpha=0.8,
            reduction="mean",
        )
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sequence, target, pileup = batch
        logits = self(sequence, pileup_embed=pileup)
        loss = self.loss_fn(
            logits.squeeze(),
            target.squeeze(),
            alpha=0.8,
            reduction="mean",
        )
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        trainable = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        optimizer = optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def parse_args(default_ablation: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SpliSync manuscript ablation."
    )
    if default_ablation is None:
        parser.add_argument(
            "--ablation",
            required=True,
            choices=ABLATIONS,
        )
    else:
        if default_ablation not in ABLATIONS:
            raise ValueError(f"Invalid default ablation: {default_ablation}")
        parser.set_defaults(ablation=default_ablation)

    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--pretrained_ckpt_dir",
        default=str(REPO_ROOT / "checkpoints"),
    )
    parser.add_argument(
        "--pretrained_name",
        default="hyenadna-medium-160k-seqlen",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "model_weights" / "ablation"),
    )
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--wandb_project")
    return parser.parse_args()


def main(default_ablation: Optional[str] = None) -> None:
    args = parse_args(default_ablation)

    tokenizer = splisync.CharacterTokenizer(
        characters=["A", "C", "G", "T"],
        model_max_length=10000 + 2,
        add_special_tokens=False,
        padding_side="left",
    )
    data_module = splisync.SeqDataModule(
        args.data_dir,
        tokenizer,
        stage="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    model = AblationLightningModule(
        ablation=args.ablation,
        pretrained_ckpt_dir=args.pretrained_ckpt_dir,
        pretrained_name=args.pretrained_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )

    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.ablation,
            log_model=False,
        )
    else:
        logger = False

    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / args.ablation
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=(
            f"{args.ablation}-{{epoch:02d}}-{{val_loss:.6f}}"
        ),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        devices="auto",
        accelerator="auto",
        strategy=(
            "ddp_find_unused_parameters_false"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=0,
        logger=logger,
        log_every_n_steps=5,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.patience,
                verbose=True,
            ),
            splisync.EvaluateCallback(),
        ],
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_ckpt or None,
    )


if __name__ == "__main__":
    main()
