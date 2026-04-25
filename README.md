PhySA-NAFNet

Official implementation of "A Physics-Informed Deep Learning Framework for Virtual Fat Suppression in Lumbar Spine MRI".

Overview

PhySA-NAFNet is a physics-informed deep learning framework for generating complementary virtual fat-suppressed contrasts from routine sagittal T1- and T2-weighted lumbar spine MRI, including a virtual STIR image and an IDEAL-like virtual water image.

Inputs

Sagittal T1-weighted image

Sagittal T2-weighted image

Outputs

Virtual STIR image

IDEAL-like virtual water image

Fat-related component

Note on Evaluation Metrics

Acquisition parameters inside the Bloch layer are initialized from the clinical protocol. They are fixed by default (learnable: false) but can optionally be set as learnable parameters.

Virtual water metrics calculated against acquired STIR are exploratory technical metrics, as acquired IDEAL water-only references were not available in this study.

Installation

conda create -n physa python=3.10
conda activate physa
pip install -r requirements.txt


Data format

See docs/data_format.md for preparing your .npy datasets and CSV splits.
To test the pipeline locally, generate dummy data:

python examples/create_dummy_data.py --out_dir examples/dummy_data


Training

python scripts/train.py --config configs/train.yaml


Inference

python scripts/infer.py --config configs/train.yaml --checkpoint checkpoints/best_G_AB.pth --input_npy examples/sample.npy --output_dir outputs/


Evaluation

python scripts/evaluate.py --pred_dir outputs/ --ref_dir data/test_stir/ --target virtual_stir


Citation

Citation will be updated after publication.
