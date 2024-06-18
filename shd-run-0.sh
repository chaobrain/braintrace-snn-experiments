for i in 100 200 300 400 600 800 1000
do
python task-image-classification.py --devices 0 --dataset shd --data_length $i --method bptt --epoch 100 --n_layer 3 --n_rec 512 --model lif-delta --exp_name test-shd --ff_scale 4.0 --rec_scale 0.5 --warmup_ratio 0.
done

