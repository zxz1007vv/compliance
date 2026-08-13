"""Command-line helpers shared by training and evaluation entry points."""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train a learning-compliance task")
    parser.add_argument("--task", default="b1_z1_ik")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "wandb", "both", "none"),
        help="Override COMPLIANCE_LOGGER for this run",
    )
    viewer = parser.add_mutually_exclusive_group()
    viewer.add_argument("--headless", dest="headless", action="store_true")
    viewer.add_argument("--viewer", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser.parse_args()
