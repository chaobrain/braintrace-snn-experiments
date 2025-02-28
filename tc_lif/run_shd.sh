# recurrent: 700-128-128-20 paras: 141.8K
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network fb --method bptt
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network fb --method d-rtrl
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network fb --method esd-rtrl --etrace_decay 0.98
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network fb --method esd-rtrl --etrace_decay 0.99


## feedforward: 700-128-128-20 paras: 108.8K
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network ff --method bptt
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network ff --method d-rtrl
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network ff --method esd-rtrl --etrace_decay 0.98
python train_shd_ssc.py --neuron tclif --task shd --beta1 1.0 --beta2 -1.0 --threshold 1.5 --gamma 0.5 --sg triangle --network ff --method esd-rtrl --etrace_decay 0.99

