## Installation
The installation instructions are based on [conda](https://conda.io/) and works on **Linux systems
only**. Therefore, please [install conda](https://conda.io/docs/install/quick.html#linux-miniconda-install) before continuing.

For installation, please download the source code of this paper and unpack it. Then, you can create a conda
environment with the following command:

```sh
$ conda env create -f environment.yml

# activate the environment
$ conda activate StylwSin

# install paper package
$ pip install ./ --no-build-isolation  
```
## Downloading the datasets
In our experiments, we use [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset for training our face reconstruction network.
Also we used [LFW](http://vis-www.cs.umass.edu/lfw/) datasets for evaluation.
All of these datasets are publicly available. To download the datasets please refer to their websites:
- [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- [LFW](http://vis-www.cs.umass.edu/lfw/)

## Running the Experiments
### Step 1: Training face reconstruction model
You can train the face reconstruction model by running `train.py`. For example, for *whitebox* attack against `ArcFace` (using `ArcFace` in loss function), you can use the following commands:
```sh
python train.py --FR_system ArcFace   --FR_loss  ArcFace  --path <path_ffhq_dataset>  
```
### Step 2: Reconstructing images
After the model is  trained, you should use it to reconstruct images.
For evaluation of a face reconstruction of restructing images, you can use the following commands:
```sh
python Reconstruct.py --ckpt <pretrained_model> --FR_system ArcFace  --path_LFW_dataset <path_LFW_dataset>
```
### Step3: Extracting features
After gettting the reconstructed images,you should extract features of those images.
You can use the following commands.
```sh
python Feature_npz.py --FR_target ArcFace --path_origin_images <origin_images> --path_Reconstructing_images <Reconstructing_images> --save_suffix <save_dir>
```

## SWFTI-pretrained models
Link: https://pan.baidu.com/s/1jQc2fy6-5ingfTFGinVF6g?pwd=mbyg 提取码: mbyg
