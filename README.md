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
- SAM weight: `sam_vit_h_4b.pth`
- Pretrained / released weight download link: [Google Drive](https://drive.google.com/file/d/13JilJT7dhxwMIgcdtnvdzr08vcbREFlR/view?usp=sharing)

## Dataset Layout

This project is configured for the following dataset structure.

ISTD:

```text
ISTD dataset
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


COD10K-v3:

```text
COD10K-v3 dataset
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


## Environment

Install runtime dependencies if needed:

```bash
cd TODO:Path to repository root
TODO:Path to Python environment/bin/pip install -r requirements.txt
```

## Run One Task

Run shadow-SAM on ISTD:

```bash
cd TODO:Path to repository root
CUDA_VISIBLE_DEVICES=0 TODO:Path to Python environment/bin/python run_eval.py --task shadow
```

Run camouflage-SAM on COD10K-v3:

```bash
cd TODO:Path to repository root
CUDA_VISIBLE_DEVICES=1 TODO:Path to Python environment/bin/python run_eval.py --task camouflage
```

## Run Both Tasks

```bash
cd TODO:Path to repository root
TODO:Path to Python environment/bin/python run_all_tests.py --shadow-gpu 0 --camouflage-gpu 1
```

Logs will be written to:

- `logs/shadow.log`
- `logs/camouflage.log`

## Expected Metrics

On the current checkpoints and datasets, the validated results are:

- `shadow-SAM / ISTD`: `shadow=2.2499`, `non_shadow=0.6068`, `ber=1.4283`
- `camouflage-SAM / COD10K-v3`: `sm=0.8726`, `em=0.8888`, `wfm=0.4057`, `mae=0.0811`
