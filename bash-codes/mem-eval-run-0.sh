for i in 50 100 200 300 400 600 800 1000
do
python task-memory-and-speed-evaluation.py --devices 0 --data_length $i --epoch 1 --n_layer 3 --n_rec 512 --method bptt --model lif-delta --exp_name mem-opt-v2 --warmup_ratio 0.
done

