# UKA-Thorax-X-Ray
UKA Thorax X-Ray preprocessing and training

## Step 1: Setup 
* Clone this repository 
* Run: `conda env create -f environment.yaml`

## Step 2: Setup Dataset
* Change `path_root` in [cxr/data/datasets/cxr_dataset.py](cxr/data/datasets/cxr_dataset.py)
* Verify dataset [tests/data/test_dataset.py](tests/data/test_dataset.py)

## Step 3: Train Model
* Run [scripts/main_train.py](scripts/main_train.py)

## Step 4: Evaluate 
* Run [scripts/main_predict.py](scripts/main_predict.py)