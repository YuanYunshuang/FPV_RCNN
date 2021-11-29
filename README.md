# FPV-RCNN: Keypoints Based Deep Feature Fusion for Cooperative Vehicle Detectionof  Autonomous  Driving
## Acknowledgement
This project is highly dependent on repo [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 
and [SpConv](https://github.com/traveller59/spconv). 

## Download dataset
* Download the [COMAP](https://seafile.cloud.uni-hannover.de/d/1c52826e98d34c0399a4/) dataset and extract all folders
* Download the pre-trained [CIA-SSD checkpoint](https://seafile.cloud.uni-hannover.de/f/35c4e520a14948eca6ef/) and store it in a new logging-path
## Setup environment
Tested on ubuntu 16.04 and cuda 10.1
```bash
apt-get update -qq && apt-get install -y software-properties-common git nano

# for compiling spconv
apt-get install -y  libboost-all-dev build-essential libssl-dev

# build python venv and install python packages
cd FPV_RCNN && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# build spconv and ops
cd spconv && python setup.py bdist_wheel
cd ./dist && pip install $(basename $./*.whl)
cd ../.. && python setup.py develop
```
## Training and test
Configurations for dataset pre-processing, model, training and testing can all be found in the python file
of folder __cfg__. To train the network with default settings, only the data path _Dataset.root_ 
and the output logging path _Optimization.PATHS['run']_  should be set. The logging path should contain 
the pre-trained CIA-SSD checkpoint.
### Training
Pass the cfg file name (ex. "fusion_pvrcnn_comap") to the function _cfg_from_py_ in the training 
or testing script, and run
```bash
tools/train_fusion_detector.py
# or
tools/test_fusion_detector.py
```
## Citation
If you find this work useful in your research, please consider cite:
```
@misc{yunshuang2021,
  title={Keypoints Based Deep Feature Fusion for Cooperative Vehicle Detectionof  Autonomous  Driving},
  author={Yunshuang Yuan, Hao Cheng and Monika Sester},
  year={2021}
}
```
or cite:
```
@crticle{comap,
AUTHOR = {Yunshuang Yuan, Monika Sester},
TITLE = {{COMAP}: A SYNTHETIC DATASET FOR COLLECTIVE MULTI-AGENT PERCEPTION OF AUTONOMOUS DRIVING},
JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {XLIII-B2-2021},
YEAR = {2021},
PAGES = {255--263},
URL = {https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B2-2021/255/2021/},
DOI = {10.5194/isprs-archives-XLIII-B2-2021-255-2021}
}
```
if you want use the COMAP dataset.




