# Description
This is the code for the journal version of [Pedestrian Trajectory Prediction Using Dynamics-based Deep Learning](https://arxiv.org/abs/2309.09021) accepted by ICRA2024.
It is something to do with the insight.

### Example
Assume there is a ETH-univ pretrained model and corresponding predicted endpoint.
```
python trainval.py --test_set eth --load_model 50 --calculate_gradient True
```


Or you can use IDE such as pycharm to modify --test_set in code manually:
```
parser.add_argument('--test_set', default='eth', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
parser.add_argument('--load_model', default=50, type=str, help="load pretrained model for test or training")
parser.add_argument('--calculate_gradient', default=True, type=ast.literal_eval, help="=True:calculate the time gradient")
```
After you modify --test_set in code manually, you can simply click 'run' 


### Reference
The relevant code heavily borrows from [TrajNet++](https://github.com/vita-epfl/trajnetplusplusbaselines)
and [velocity](https://ieeexplore.ieee.org/abstract/document/8972605).


