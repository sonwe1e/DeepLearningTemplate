import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_wandb", type=bool, default=True)
    parser.add_argument("--project", type=str, default="")

    # models

    # augmentations

    # dataset
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--image_size", type=int, default=224)

    # training setups
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)

    # experiment
    parser.add_argument("--devices", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--val_check", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    return parser.parse_args()


def get_option():
    opt = parse_args()
    return opt
