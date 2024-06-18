for i in 50 100 200 300 400 600 800 1000
do
python task-memory-and-speed-evaluation.py --devices 3 --data_length $i --epoch 1 --n_layer 3 --n_rec 512 --method expsm_diag --etrace_decay 0.9 --model lif-delta --memory_eval 1 --drop_last 1 --exp_name mem-opt-v2  --warmup_ratio 0.
done

