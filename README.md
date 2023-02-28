# SoftZoo: A Soft Robot Co-design Benchmark For Locomotion In Diverse Environments

### [[OpenReview]](https://openreview.net/forum?id=Xyme9p1rpZw)[[Demo]](https://sites.google.com/view/softzoo-iclr-2023) ###

SoftZoo is a soft robot co-design platform for locomotion in diverse environments that,
* supports an extensive, naturally-inspired material set, including the ability to simulate environments such as flat ground, desert, wetland, clay, ice, snow, shallow water, and ocean.
* provides a variety of tasks relevant for soft robotics, including fast locomotion, agile turning, and path following, as well as differentiable design representations for morphology and control.

With this platform, we study a wide variety of prevalent representations and co-design algorithms.

![pipeline](imgs/demo.png)

## Dependencies
### (Recommended) Create conda environment
```
$ conda create -n softzoo python==3.8
$ conda activate softzoo
```
### Install required packages.
(other versions may also work but we tested with the listed versions).
```
$ pip install taichi==1.4.1
$ conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install -r requirments.txt
```
### Other packages.
(only required in some of the experiments/tools; also while we only tested with the following versions, other versions may still work).
* [mesh_to_sdf-0.0.14](https://github.com/marian42/mesh_to_sdf)
* [pynput-1.7.6](https://pypi.org/project/pynput/)
* [seaborn-0.12.1](https://pypi.org/project/seaborn/)
* [stable-baselines3-1.4.0](https://stable-baselines3.readthedocs.io/en/master/)
* [huggingface-sb3-2.2.4](https://pypi.org/project/huggingface-sb3/)
* [optuna-3.0.0](https://pypi.org/project/optuna/)
### Install SoftZoo.
```
$ git clone git@github.com:zswang666/softzoo.git
$ cd softzoo/
$ pip install -e .
```

## (TODO) Download Assets
Mention where the assets are from.

## Running Experiments
### Control optimization with differentiable physics.
Use differentiable physics to optimize the control of an animal-like robot with human-annotated muscle placement and direction. You can try out different animals in different environments. You can also run with customized robot design (in pcd format) via this script.
```
$ bash scripts/diffsim_with_annotated_pcd.sh 
```
### Control optimization with reinforcement learning
Use reinforcement to optimize the control of an animal-like robot with human-annotated muscle placement and direction. We adapt implementation from `stable-baselines3` and use PPO for the experiments in the paper.

Additionally required packages: seaborn, stable-baselines3, optuna, huggingface_sb3
```
$ bash scripts/rl_with_annotated_pcd.sh 
```

### (TODO) Co-design / design optimization with differentiable physics.

### (TODO) Co-design with evolutionary algorithm.

### Demo example of extension to manipulation.
Demonstrate a simple example that uses differentiable physics to optimize a hand-like soft robot to throw a snowball.
```
$ bash scripts/demo_manipulation.sh
```

### Human control.
We provide a simple interactive script that allows human control of the soft robot. This script is particularly useful to check if a robot design is even controllable. By default, we use keys `['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'z', 'c', 'b']` for positive force and keys `['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'Return', 'x', 'v', 'n']` for negative force. Check more details in [algorithms/human/run.py](algorithms/human/run.py). Note that this script requires on-screen rendering.
```
$ python -m algorithms.human.run --env-config-file human.yaml --designer-type annotated_pcd --annotated-pcd-path ./softzoo/assets/meshes/pcd/Caterpillar.pcd
```

## Tools
### Manual annotation of muscle placement.
We provide a simple script to convert a 3D mesh into points (which can be consumed by SoftZoo, or more precisely MPM) along with functions to manually annotate muscle group and automatically generate muscle direction. Note that this script requires on-screen rendering.
```
$ python -m softzoo.tools.annotate_mesh --mesh-path ./softzoo/assets/meshes/stl/Hippo.stl --out-path ./softzoo/assets/meshes/pcd/Hippo.pcd --n-clusters 5
```

### (TODO) Generate terrain texture.


## (TODO) Render with GL
