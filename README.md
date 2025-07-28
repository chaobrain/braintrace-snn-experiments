# ``BrainScale`` experiments on spiking neural networks




## Requirements

For the main computation of the project, we use the following packages:

- [brainunit](https://github.com/chaobrain/brainunit)
- [brainstate](https://github.com/chaobrain/brainstate)
- [brainscale](https://github.com/chaobrain/brainscale)
- [braintools](https://github.com/chaobrain/braintools)

For the dataset generation, we use the following packages:

- tonic
- pytorch
- torchvision

For the checkpoint and logging of the model parameters, we use the following packages:

- orbax.checkpoint



## RSNN long-term dependency evaluation: DMS task

```bash

# BPTT
python task-rsnn-long-term-dependency.py --epochs 2000 --method bptt --dataset dms --t_delay 1000 \
  --tau_I2 1500 --tau_neu 100 --tau_syn 100 --n_rec 200 --lr 0.001 --A2 1 --optimizer adam --devices 3 \
  --t_fixation 10. --spk_fun relu --acc_th 0.95 --n_data_worker 4  --dt 1. --ff_scale 6 --rec_scale 2
  
  
# ES-D-RTRL (IO Dim)  
python task-rsnn-long-term-dependency.py --epochs 2000 --method expsm_diag --etrace_decay 0.99 --vjp_time t --dataset dms --t_delay 1000 \
    --tau_I2 1500 --tau_neu 100 --tau_syn 100 --n_rec 200 --lr 0.001 --A2 1  --optimizer adam --devices 3 \
    --t_fixation 10. --spk_fun relu --acc_th 0.95  --n_data_worker 4  --dt 1. --ff_scale 6 --rec_scale 2
    

# D-RTRL (Param Dim)
python task-rsnn-long-term-dependency.py --epochs 2000   --method diag --dataset dms  --t_delay 1000 \
    --tau_I2 1500 --tau_neu 100 --tau_syn 100 --n_rec 200 --lr 0.001 --A2 1   --optimizer adam --devices 3 \
    --t_fixation 10. --spk_fun relu --acc_th 0.95  --n_data_worker 4  --dt 1. --ff_scale 6 --rec_scale 2

```



## EI network for decision making tasks

ES-D-RTRL training of the EI network for decision making tasks. 

```bash
cd ./ei_coba_net_decision_making
python training.py --tau_neu 200 --tau_syn 10 --tau_I2 2000  --ff_scale 4.0 --rec_scale 2.0  --method esd-rtrl  --n_rec 800  --epoch_per_step 20 --diff_spike  0  --epochs 300 --lr 0.001 --etrace_decay 0.9
```

D-RTRL training of the EI network for decision making tasks. 

```bash
python training.py --tau_neu 200 --tau_syn 10 --tau_I2 2000  --ff_scale 4.0 --rec_scale 2.0  --method d-rtrl  --n_rec 800  --epoch_per_step 30 --diff_spike  0  --epochs 300 --lr 0.001
```


BPTT training of the EI network for decision making tasks. 

```bash
python training.py --tau_neu 200 --tau_syn 10 --tau_I2 2000  --ff_scale 4.0 --rec_scale 2.0  --method bptt  --n_rec 800  --epoch_per_step 30 --diff_spike  0  --epochs 300 --lr 0.001 
```


## Memory and speed evaluation


```bash

python task-memory-and-speed-evaluation-tpu.py

```





## RSNN image classification on Gesture dataset

The code below is used to train a spiking neural network on the Gesture dataset using different methods (BPTT, ES-D-RTRL, D-RTRL).

The codebase is located in `./event_gru_dvs_gesture` di


BPTT

```bash
python main.py --batch-size 64 --units 1024 \
    --num-layers 1 --frame-size 128 --method bptt  \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 100 \
    --event-agg-method mean --use-cnn --dropout 0.5 --zoneout 0 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.25 --augment-data \
    --devices 0  --data ../data --cache ./cache
```


D-RTRL

```bash
python main.py --batch-size 64 --units 1024 \
    --num-layers 1 --frame-size 128 --method d-rtrl  \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 50 \
    --event-agg-method mean --use-cnn --dropout 0.5 --zoneout 0 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.25 \
    --augment-data  --data ../data --cache ./cache --devices 1
```

ES-D-RTRL

```bash
python main.py --batch-size 64 --units 1024 \
    --num-layers 1 --frame-size 128 --method es-d-rtrl --etrace-decay 0.2  \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 100 \
    --event-agg-method mean --use-cnn --dropout 0.5 --zoneout 0 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.25 \
    --augment-data  --data ../data --cache ./cache --devices 6
```



## RSNN classification on SHD dataset


The codebase is located in `./sparch` directory. The code below is used to train a spiking neural network on the SHD dataset using different methods (BPTT, ES-D-RTRL, D-RTRL).

BPTT

```bash
python main.py --model_type LIF --dataset_name shd  --nb_epochs 100 --method bptt --nb_hiddens 1024
python main.py --model_type RLIF --dataset_name shd  --nb_epochs 100 --method bptt --nb_hiddens 1024
python main.py --model_type adLIF --dataset_name shd --nb_epochs 100 --method bptt --nb_hiddens 1024
python main.py --model_type RadLIF --dataset_name shd --nb_epochs 100 --method bptt --nb_hiddens 1024
```

D-RTRL

```bash
python main.py --model_type LIF --dataset_name shd  --nb_epochs 100 --method d-rtrl --nb_hiddens 1024
python main.py --model_type RLIF --dataset_name shd  --nb_epochs 100 --method d-rtrl --nb_hiddens 1024
python main.py --model_type adLIF --dataset_name shd --nb_epochs 100 --method d-rtrl --nb_hiddens 1024
python main.py --model_type RadLIF --dataset_name shd --nb_epochs 100 --method d-rtrl --nb_hiddens 1024
```


ES-D-RTRL

```bash
python main.py --model_type LIF --dataset_name shd --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --etrace_decay 0.8
python main.py --model_type RLIF --dataset_name shd --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --etrace_decay 0.8
python main.py --model_type adLIF --dataset_name shd --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --etrace_decay 0.98
python main.py --model_type RadLIF --dataset_name shd --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --etrace_decay 0.98
```

## Citations

```text
@article {Wang2024.09.24.614728,
	author = {Wang, Chaoming and Dong, Xingsi and Ji, Zilong and Jiang, Jiedong and Liu, Xiao and Wu, Si},
	title = {BrainScale: Enabling Scalable Online Learning in Spiking Neural Networks},
	elocation-id = {2024.09.24.614728},
	year = {2025},
	doi = {10.1101/2024.09.24.614728},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/07/27/2024.09.24.614728},
	eprint = {https://www.biorxiv.org/content/early/2025/07/27/2024.09.24.614728.full.pdf},
	journal = {bioRxiv}
}
```

