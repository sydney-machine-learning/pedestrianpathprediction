# Description
This is the code for [Pedestrian Trajectory Prediction Using Dynamics-based Deep Learning](https://arxiv.org/abs/2309.09021), which is under review.

If you find this code useful then please cite:
```bibtex
@article{wang2023pedestrian,
title={Pedestrian Trajectory Prediction Using Dynamics-based Deep Learning},
author={Wang, Honghui and Zhi, Weiming and Batista, Gustavo and Chandra, Rohitash},
journal={arXiv preprint arXiv:2309.09021},
year={2023}
}
```

### Environment
The version of python is 3.8.16.

```bash
pip install numpy==1.24.3
pip install torch==1.7.0
pip install pyyaml==6.0
pip install tqdm==4.65.0
pip install scipy==1.10.1
pip install -U scikit-learn==1.2.2
pip install numba==0.57.0
```


### Train
The Default settings are to train on ETH-univ dataset. 

Data cache and models will be stored in the subdirectory "./output/eth/" by default. Notice that for this repo, we only provide implementation on GPU, except for endpoint estimation.

```
git clone https://github.com/sydney-machine-learning/pedestrianpathprediction.git
cd pedestrianpathprediction
python trainval.py --test_set <dataset to evaluate>
```

The datasets are selected on arguments '--test_set'. Five datasets in ETH/UCY including
[**eth, hotel, zara1, zara2, univ**].

### Example

This command is to train model for ETH-univ. For different dataset, change 'hotel' to other datasets named in the last section.

```
python trainval.py --test_set eth
```

Or you can use IDE such as pycharm to modify --test_set in code manually:
```
parser.add_argument('--test_set', default='eth', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
```
After you modify --test_set in code manually, you can simply click 'run' to train.



### Acknowledgements
This research includes computations using the computational cluster Katana (doi: https://doi.org/10.26190/669x-a286) supported by Research Technology Services at UNSW Sydney.


### Reference
The code base heavily borrows from [STAR](https://github.com/Majiker/STAR), and goal estimator we use heavily borrows from [Heading](https://github.com/JoeHEZHAO/expert_traj). 