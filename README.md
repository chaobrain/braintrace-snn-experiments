# ``BrainScale`` experiments on spiking neural networks




## Requirements

For the main computation of the project, we use the following packages:

- brainunit
- brainstate
- brainscale
- braintools
- brainpy-dataset

For the dataset generation, we use the following packages:

- tonic
- pytorch
- torchvision
- orbax.checkpoint



## RSNN long-term dependency evaluation: DMS task

```bash
# BPTT
python task-rsnn-long-term-dependency.py --epochs 2000 --mode train --method bptt --dataset dms \
  --t_delay 1500 --tau_I2 3000 --n_rec 200 --lr 0.001 --A2 -0.1 --optimizer adam --devices 3 \
  --t_fixation 10. --spk_fun multi_gaussian --acc_th 1.
  
  
# ES-D-RTRL (IO Dim)  
python task-rsnn-long-term-dependency.py --epochs 1000 --mode train --method expsm_diag --diag_jacobian exact \
  --etrace_decay 0.98 --vjp_time t --dataset dms --t_delay 1500 --tau_I2 3000 --n_rec 200 \
  --lr 0.001 --A2 -0.1 --optimizer adam --devices 1 --t_fixation 10. --spk_fun multi_gaussian --acc_th 0.95


# D-RTRL (Param Dim)
python task-rsnn-long-term-dependency.py --epochs 1000 --mode train --method diag --diag_jacobian exact \
  --vjp_time t --dataset dms --t_delay 1500 --tau_I2 3000 --n_rec 200 --lr 0.001 --A2 -0.1 \
  --optimizer adam --devices 2 --t_fixation 10. --spk_fun multi_gaussian --acc_th 0.95
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






