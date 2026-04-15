import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = {
    "shadow": PROJECT_ROOT / "configs" / "shadow_istd_vith_eval.yaml",
    "camouflage": PROJECT_ROOT / "configs" / "camouflage_cod10k_vith_eval.yaml",
}


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_loader(config: dict, num_workers: int) -> DataLoader:
    spec = config["test_dataset"]
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    return DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
    )


def resolve_metric(eval_type: str):
    if eval_type == "f1":
        return utils.calc_f1, ("f1", "auc", "none", "none")
    if eval_type == "fmeasure":
        return utils.calc_fmeasure, ("f_mea", "mae", "none", "none")
    if eval_type == "ber":
        return utils.calc_ber, ("shadow", "non_shadow", "ber", "none")
    if eval_type == "cod":
        return utils.calc_cod, ("sm", "em", "wfm", "mae")
    raise ValueError(f"Unsupported eval_type: {eval_type}")


def evaluate(loader: DataLoader, model, eval_type: str, use_amp: bool) -> dict:
    metric_fn, metric_names = resolve_metric(eval_type)
    meters = [utils.Averager() for _ in range(4)]

    model.eval()
    progress = tqdm(loader, leave=False, desc="eval")
    for batch in progress:
        for key, value in batch.items():
            batch[key] = value.cuda(non_blocking=True)

        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = torch.sigmoid(model.infer(batch["inp"]))
            else:
                pred = torch.sigmoid(model.infer(batch["inp"]))

        pred = pred.float()
        results = metric_fn(pred, batch["gt"])
        batch_size = batch["inp"].shape[0]

        for meter, result in zip(meters, results):
            meter.add(float(result), batch_size)

        progress.set_description(
            "eval {} {:.4f} | {} {:.4f} | {} {:.4f} | {} {:.4f}".format(
                metric_names[0], meters[0].item(),
                metric_names[1], meters[1].item(),
                metric_names[2], meters[2].item(),
                metric_names[3], meters[3].item(),
            )
        )

        del pred
        torch.cuda.empty_cache()

    return {name: meter.item() for name, meter in zip(metric_names, meters)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=sorted(DEFAULT_CONFIGS.keys()),
        help="Run a predefined evaluation task.",
    )
    parser.add_argument(
        "--config",
        help="Run with a specific YAML config. Overrides --task when provided.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable fp16 autocast. Not recommended for ViT-H on 24GB GPUs.",
    )
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config).resolve()
    elif args.task:
        config_path = DEFAULT_CONFIGS[args.task]
    else:
        parser.error("Either --task or --config must be provided.")

    config = load_config(config_path)
    model_checkpoint = config["model_checkpoint"]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"Using config: {config_path}")
    print(f"Using checkpoint: {model_checkpoint}")
    print(f"Using GPU: {args.gpu}")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    loader = build_loader(config, num_workers=args.num_workers)
    model = models.make(config["model"]).cuda()
    state_dict = torch.load(model_checkpoint, map_location="cuda:0")
    model.load_state_dict(state_dict, strict=True)

    metrics = evaluate(
        loader=loader,
        model=model,
        eval_type=config["eval_type"],
        use_amp=not args.disable_amp,
    )

    print("Final metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
