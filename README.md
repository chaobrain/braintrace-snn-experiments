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



## Memory and speed evaluation


```bash

python task-memory-and-speed-evaluation-tpu.py

```





## RSNN image classification on N-MNIST dataset


```bash
# BPTT
python task-image-classification.py --devices 0 --dataset nmnist --data_length 400 --epoch 100 --n_layer 3 --n_rec 512 --method bptt --model lif-delta --exp_name test-nmnist --warmup_ratio 0.

# ES-D-RTRL (IO Dim)
python task-image-classification.py --devices 1 --dataset nmnist --data_length 400 --epoch 100 --n_layer 3 --n_rec 512 --method expsm_diag --etrace_decay 0.9 --model lif-delta --exp_name test-nmnist --warmup_ratio 0.

# D-RTRL (Param Dim)
python task-image-classification.py --devices 2 --dataset nmnist --data_length 400 --epoch 100 --n_layer 3 --n_rec 512 --method diag --model lif-delta --exp_name test-nmnist --warmup_ratio 0.
```


## RSNN image classification on SHD dataset


```bash
# BPTT
python task-image-classification.py --devices 0 --dataset shd --data_length 400 --epoch 100 --n_layer 3 --n_rec 512 --method bptt --model lif-delta --exp_name test-shd --warmup_ratio 0.

# ES-D-RTRL (IO Dim)
python task-image-classification.py --devices 1 --dataset shd --data_length 400 --epoch 100 --n_layer 3 --n_rec 512 --method expsm_diag --etrace_decay 0.9 --model lif-delta --exp_name test-shd --warmup_ratio 0.

# D-RTRL (Param Dim)
python task-image-classification.py --devices 2 --dataset shd --data_length 400 --epoch 100 --n_layer 3 --n_rec 512 --method diag --model lif-delta --exp_name test-shd --warmup_ratio 0.
```



