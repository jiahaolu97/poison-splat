# Poison-splat

## Project Organization
```
poison-splat
    |---attacker
        (code files for attacker to create poisoned data)
    |---dataset
        (folder for saving clean and poisoned data)
    |---exp
        (code files for ablation studies and visualizations)
    |---log
        (folder for saving experiment results)
    |---utils
        (useful tools)
    |---victim
        (code files for black-box victim GS systems)
```

## Installation and Quick Start
### Install environment
First create a conda environment with pytorch-gpu. CUDA version 11.8 recommended:
```
conda create -n poison_splat python=3.11 -y
conda activate poison_splat
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### Download clean datasets

Download the datasets following instructions from `dataset` folder. You can verify your installation successful after `Nerf_Synthetic` are downloaded by running:
```
bash exp/00_test/test_install.sh
```

### Benchmark clean datasets for victim
```

```

### Create poisoned datasets


### Benchmark poisoned datasets for victim
