#!/usr/bin/env python3
"""Train SpliSync with a frozen randomly initialized HyenaDNA backbone."""

from train_ablation import main


if __name__ == "__main__":
    main("random_hyena")
