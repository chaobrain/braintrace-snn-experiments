for i in 100 200 300 400 600 800 1000
do
python task-image-classification.py --devices 3 --dataset shd --data_length $i --epoch 100 --n_layer 3 --n_rec 512 --method diag --model lif-delta --exp_name test-shd --ff_scale 4.0 --rec_scale 0.5 --warmup_ratio 0.
done

