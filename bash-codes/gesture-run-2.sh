for i in 50 100 200 300 400 600 800 1000
do
python task-image-classification.py --devices 2 --data_length $i --epoch 100 --n_layer 3 --n_rec 512 --method diag --model lif-delta --exp_name test1 --warmup_ratio 0.
done
