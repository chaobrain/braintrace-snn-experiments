# ``BrainScale`` experiments on spiking neural networks




## Requirements

For the main computation of the project, we use the following packages:

- [brainunit](https://github.com/chaobrain/brainunit)
- [brainstate](https://github.com/chaobrain/brainstate)
- [brainscale](https://github.com/chaobrain/brainscale)
- [braintools](https://github.com/chaobrain/braintools)
- brainpy-dataset

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



## RNN long-term dependency evaluation: copying task


Change the sequence length using the `--data_length` argument. Here is an example of training with a sequence length of 500.


```bash

# BPTT training for copying task
python task-rnn-long-term-dependency.py --dataset copying --batch_size 128 --lr 1e-3 --devices 0  \
  --dt 1.0 --method bptt --model gru  --data_length 500  --n_data_worker 10 --epochs 100000 --loss cel

# Diag training for copying task
python task-rnn-long-term-dependency.py --dataset copying --batch_size 128 --lr 1e-3 --devices 1  \
  --dt 1.0 --method diag --model gru  --data_length 500  --n_data_worker 10 --epochs 100000 --loss cel
  
```


## Memory and speed evaluation


```bash

# BPTT
for i in 50 100 200 300 400 600 800 1000
do
python task-memory-and-speed-evaluation.py --devices 0 --data_length $i --epoch 1 --n_layer 3 --n_rec 512 --method bptt --model lif-delta 
done


# ES-D-RTRL (IO Dim)  
for i in 50 100 200 300 400 600 800 1000
do
python task-memory-and-speed-evaluation.py --devices 2 --data_length $i --epoch 1 --n_layer 3 --n_rec 512 --method expsm_diag --etrace_decay 0.9 --model lif-delta --memory_eval 1 --drop_last 1
done


# D-RTRL (Param Dim)
for i in 50 100 200 300 400 600 800 1000
do
python task-memory-and-speed-evaluation.py --devices 2 --data_length $i --epoch 1 --n_layer 3 --n_rec 512 --method diag --model lif-delta --memory_eval 1 --drop_last 1
done

```



## RSNN image classification on DVS Gesture dataset

Firstly, we should preprocess the dataset using the following command:


```bash
python dvs-gesture-preprocessing.py
```

Training based on BPTT:

```bash
for i in 50 100 200 300 400 600 800 1000
do
python task-image-classification.py --devices 0 --dataset gesture --data_length $i --epoch 100 --n_layer 3 --n_rec 512 --method bptt --model lif-delta --exp_name test1 --warmup_ratio 0.
done
```

Training based on ES-D-RTRL (IO Dim):

```bash
for i in 50 100 200 300 400 600 800 1000
do
python task-image-classification.py --devices 1 --dataset gesture --data_length $i --epoch 100 --n_layer 3 --n_rec 512 --method expsm_diag --etrace_decay 0.9 --model lif-delta --exp_name test1  --warmup_ratio 0.
done
```

Training based on D-RTRL (Param Dim):

```bash
for i in 50 100 200 300 400 600 800 1000
do
python task-image-classification.py --devices 2 --dataset gesture --data_length $i --epoch 100 --n_layer 3 --n_rec 512 --method diag --model lif-delta --exp_name test1 --warmup_ratio 0.
done
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



## Conductance-based Excitatory and Inhibitory RSNN on Evidence Accumulation Task


```bash
# Training with BPTT
python task-coba-ei-rsnn.py --method bptt


# Training with D-RTRL
python task-coba-ei-rsnn.py --method diag

python task-coba-ei-rsnn.py --tau_neu 100 --tau_syn 50 --tau_a 1500 --ff_scale 0.4 --rec_scale 0.1 --vjp_time t_minus_1



# Training with ES-D-RTRL
python task-coba-ei-rsnn.py --method expsm_diag --etrace_decay 0.98




python task-coba-ei-rsnn.py --tau_neu 100 --tau_syn 10 --tau_a 1500 --mode sim --ff_scale 0.5 --rec_scale 0.1 




python task-coba-ei-rsnn.py --tau_neu 100 --tau_syn 10 --tau_a 2500 --mode sim --ff_scale 10.0 --rec_scale 2. --net cuba



python task-coba-ei-rsnn.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method diag --diag_normalize 0 --vjp_time t_minus_1 --n_rec 400


python task-coba-ei-rsnn.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --vjp_time t_minus_1 --n_rec 400


python task-coba-ei-rsnn.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --etrace_decay 0.98 --vjp_time t_minus_1 --n_rec 400
python task-coba-ei-rsnn.py --devices 2 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --etrace_decay 0.98 --vjp_time t_minus_1 --n_rec 400

python task-coba-ei-rsnn.py --devices 1 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method diag --diag_normalize 0 --vjp_time t --n_rec 400

python task-coba-ei-rsnn.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method bptt --n_rec 400




python task-coba-ei-rsnn.py --devices 2 --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --etrace_decay 0.98 --vjp_time t_minus_1 --n_rec 400

```







