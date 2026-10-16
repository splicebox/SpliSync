#!/usr/bin/env python3
"""Run inference with alignment-only, sequence-only, or one-hot CNN models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import train_ablation


SUPPORTED_ABLATIONS = (
    "alignment_only",
    "sequence_only",
    "onehot_cnn",
)
ABLATION_ALIASES = {
    "cnn_sequence": "onehot_cnn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SpliSync inference with an alignment-only, sequence-only, "
            "or one-hot CNN ablation checkpoint."
        )
    )
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument(
        "--ablation",
        choices=("auto", *SUPPORTED_ABLATIONS),
        default="auto",
    )
    parser.add_argument(
        "--pretrained_ckpt_dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--pretrained_name",
        default="hyenadna-medium-160k-seqlen",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a CUDA device such as cuda:0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Prediction TSV (default: DATA_DIR.predictions.tsv)",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def resolve_ablation(requested: str, hyperparameters: dict) -> str:
    checkpoint_ablation = hyperparameters.get("ablation")
    checkpoint_ablation = ABLATION_ALIASES.get(
        checkpoint_ablation,
        checkpoint_ablation,
    )
    if requested == "auto":
        if checkpoint_ablation not in SUPPORTED_ABLATIONS:
            raise ValueError(
                "Checkpoint does not identify a supported component "
                "ablation; pass --ablation explicitly."
            )
        return checkpoint_ablation
    if checkpoint_ablation and checkpoint_ablation != requested:
        raise ValueError(
            "Requested ablation conflicts with checkpoint metadata: "
            f"requested={requested!r}, checkpoint={checkpoint_ablation!r}"
        )
    return requested


def main() -> None:
    args = parse_args()
    checkpoint_path = args.model_path.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Inference data directory not found: {data_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint has no state_dict: {checkpoint_path}")
    hyperparameters = checkpoint.get("hyper_parameters", {})
    ablation = resolve_ablation(args.ablation, hyperparameters)

    model = train_ablation.AblationLightningModule(
        ablation=ablation,
        pretrained_ckpt_dir=str(args.pretrained_ckpt_dir.expanduser().resolve()),
        pretrained_name=hyperparameters.get(
            "pretrained_name",
            args.pretrained_name,
        ),
        lr=hyperparameters.get("lr", 1e-3),
        weight_decay=hyperparameters.get("weight_decay", 1e-2),
        max_epochs=hyperparameters.get("max_epochs", 100),
    )
    model.load_state_dict(state_dict, strict=True)

    tokenizer = train_ablation.splisync.CharacterTokenizer(
        characters=["A", "C", "G", "T"],
        model_max_length=10000 + 2,
        add_special_tokens=False,
        padding_side="left",
    )
    data_module = train_ablation.splisync.SeqDataModule(
        str(data_dir),
        tokenizer,
        stage="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    data_module.setup("test")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else Path(f"{data_dir}.predictions.tsv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model.to(device)
    result = train_ablation.splisync.evaluate(
        model,
        data_module.test_dataloader(),
        model.loss_fn,
        device,
        stage="test",
        csv_path=str(output_path),
    )

    print(f"Ablation: {ablation}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Predictions: {result['csv_path']}")
    print(f"Predicted sites: {result['csv_rows']}")


if __name__ == "__main__":
    main()
