# Conductance-based Excitatory and Inhibitory RSNN on Evidence Accumulation Task


```bash
# Training with BPTT
python training.py --method bptt

# Training with D-RTRL
python training.py --method diag

python training.py --tau_neu 100 --tau_syn 50 --tau_a 1500 --ff_scale 0.4 --rec_scale 0.1 --vjp_time t_minus_1

# Training with ES-D-RTRL
python training.py --method expsm_diag --etrace_decay 0.98

python training.py --tau_neu 100 --tau_syn 10 --tau_a 1500 --mode sim --ff_scale 0.5 --rec_scale 0.1 

python training.py --tau_neu 100 --tau_syn 10 --tau_a 2500 --mode sim --ff_scale 10.0 --rec_scale 2. --net cuba

python training.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method diag --diag_normalize 0 --vjp_time t_minus_1 --n_rec 400

python training.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --vjp_time t_minus_1 --n_rec 400

python training.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --etrace_decay 0.98 --vjp_time t_minus_1 --n_rec 400
python training.py --devices 2 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --etrace_decay 0.98 --vjp_time t_minus_1 --n_rec 400

python training.py --devices 1 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method diag --diag_normalize 0 --vjp_time t --n_rec 400

python training.py --devices 0 --tau_neu 100 --tau_syn 10 --tau_a 2500  --ff_scale 1.0 --rec_scale 0.5 --method bptt --n_rec 400

python training.py --devices 2 --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method expsm_diag --diag_normalize 0 --etrace_decay 0.98 --vjp_time t_minus_1 --n_rec 400

```





```bash
python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method d-rtrl   --n_rec 400

python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method esd-rtrl --etrace_decay 0.98  --n_rec 400

```



