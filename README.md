# SAM-Adapter Shadow and Camouflage Evaluation

This project is a standalone evaluation-only package for:

- `shadow-SAM` on `ISTD`
- `camouflage-SAM` on `COD10K-v3`

This repository is a supplement to [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch).
Thanks to the authors for this awesome work.

It is extracted from `SAM-Adapter-PyTorch` and reduced to the minimum code needed
to run testing independently. Dataset paths and checkpoint paths remain unchanged.

## Project Layout

```text
SAM-Adapter_Shadow_and_Camouflages/
├── configs/
│   ├── camouflage_cod10k_vith_eval.yaml
│   └── shadow_istd_vith_eval.yaml
├── datasets/
├── models/
├── logs/
├── run_all_tests.py
├── run_eval.py
├── requirements.txt
├── sod_metric.py
└── utils.py
```

## Upstream Project and Weights

- Upstream repository: [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)
- Pretrained / released weight download link: [Google Drive](https://drive.google.com/file/d/13JilJT7dhxwMIgcdtnvdzr08vcbREFlR/view?usp=sharing)

## Dataset Layout

This project is configured for the following dataset structure.

ISTD:

```text
/share/test/sunye_2/zhangchuhan/ISTD/ISTD_Dataset/
└── test/
    ├── test_A/
    │   ├── 100-1.png
    │   ├── 100-2.png
    │   └── ...
    ├── test_B/
    │   ├── 100-1.png
    │   ├── 100-2.png
    │   └── ...
    └── test_C/
        ├── 100-1.png
        ├── 100-2.png
        └── ...
```

This evaluation project uses:

- image path: `/share/test/sunye_2/zhangchuhan/ISTD/ISTD_Dataset/test/test_A`
- mask path: `/share/test/sunye_2/zhangchuhan/ISTD/ISTD_Dataset/test/test_B`

COD10K-v3:

```text
/share/test/sunye_2/zhangchuhan/COD/COD10K-v3/
├── Info/
└── Test/
    ├── Image/
    │   ├── COD10K-NonCAM-1-Amphibian-3.jpg
    │   ├── COD10K-NonCAM-1-Amphibian-4.jpg
    │   └── ...
    ├── GT_Object/
    │   ├── COD10K-NonCAM-1-Amphibian-3.png
    │   ├── COD10K-NonCAM-1-Amphibian-4.png
    │   └── ...
    ├── GT_Edge/
    │   ├── COD10K-NonCAM-1-Amphibian-3.png
    │   ├── COD10K-NonCAM-1-Amphibian-4.png
    │   └── ...
    ├── CAM_Instance_Test.json
    └── CAM-NonCAM_Instance_Test.txt
```

This evaluation project uses:

- image path: `/share/test/sunye_2/zhangchuhan/COD/COD10K-v3/Test/Image`
- mask path: `/share/test/sunye_2/zhangchuhan/COD/COD10K-v3/Test/GT_Object`

## Environment

Recommended environment:

```bash
/home/sunye_2/zhangchuhan/sam3_env
```

Install runtime dependencies if needed:

```bash
cd /home/sunye_2/zhangchuhan/git_repo/SAM-Adapter_Shadow_and_Camouflages
/home/sunye_2/zhangchuhan/sam3_env/bin/pip install -r requirements.txt
```

## Important Note

The provided fine-tuned checkpoints are `ViT-H` checkpoints:

- `/home/sunye_2/zhangchuhan/best_results/best_results/istd/model_epoch_best.pth`
- `/home/sunye_2/zhangchuhan/best_results/best_results/cod/model_epoch_best.pth`

So this standalone project is configured for `ViT-H` evaluation. It does not use
the `sam_vit_b_01ec64.pth` base checkpoint, because that checkpoint is not
compatible with the actual fine-tuned model shapes.

Mixed precision is enabled by default in `run_eval.py` to avoid OOM on `24 GB`
GPUs when evaluating `ViT-H` at `1024 x 1024`.

## Run One Task

Run shadow-SAM on ISTD:

```bash
cd /home/sunye_2/zhangchuhan/git_repo/SAM-Adapter_Shadow_and_Camouflages
CUDA_VISIBLE_DEVICES=0 /home/sunye_2/zhangchuhan/sam3_env/bin/python run_eval.py --task shadow
```

Run camouflage-SAM on COD10K-v3:

```bash
cd /home/sunye_2/zhangchuhan/git_repo/SAM-Adapter_Shadow_and_Camouflages
CUDA_VISIBLE_DEVICES=1 /home/sunye_2/zhangchuhan/sam3_env/bin/python run_eval.py --task camouflage
```

## Run Both Tasks

```bash
cd /home/sunye_2/zhangchuhan/git_repo/SAM-Adapter_Shadow_and_Camouflages
/home/sunye_2/zhangchuhan/sam3_env/bin/python run_all_tests.py --shadow-gpu 0 --camouflage-gpu 1
```

Logs will be written to:

- `logs/shadow.log`
- `logs/camouflage.log`

## Expected Metrics

On the current checkpoints and datasets, the validated results are:

- `shadow-SAM / ISTD`: `shadow=2.2499`, `non_shadow=0.6068`, `ber=1.4283`
- `camouflage-SAM / COD10K-v3`: `sm=0.8726`, `em=0.8888`, `wfm=0.4057`, `mae=0.0811`
