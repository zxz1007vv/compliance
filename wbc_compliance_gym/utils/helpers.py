"""Command-line helpers shared by training and evaluation entry points."""

import argparse

from wbc_compliance_gym.envs import DEFAULT_TASK


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a registered whole-body compliance task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "examples:\n"
            f"  python scripts/train.py --task {DEFAULT_TASK}\n"
            f"  python scripts/train.py --task {DEFAULT_TASK} --resume\n"
            "  python scripts/train.py --resume-run-dir logs/<task>/<run> "
            f"--task {DEFAULT_TASK} --checkpoint latest"
        ),
    )
    task_selection = parser.add_mutually_exclusive_group(required=True)
    task_selection.add_argument(
        "--task", help="Registered task to train"
    )
    task_selection.add_argument(
        "--list-tasks", action="store_true", help="List registered tasks and exit"
    )
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the selected task from its latest local run",
    )
    parser.add_argument(
        "--resume-run-dir",
        help="Resume from this run; implies --resume and overrides latest-run selection",
    )
    parser.add_argument(
        "--checkpoint", default="latest", help="Checkpoint iteration or 'latest'"
    )
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
