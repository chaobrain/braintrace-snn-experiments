

cd ..

for run in {0..5}
do
for lr in 0.02 0.01 0.005 0.001
do
  python main.py --model_type LIF --dataset_name shd  --nb_epochs 200 --method bptt --nb_hiddens 1024 --lr $lr --devices 0  --inp_scale 10. --rec_scale 5. --nb_layers 2
  python main.py --model_type RLIF --dataset_name shd  --nb_epochs 200 --method bptt --nb_hiddens 1024 --lr $lr --devices 0  --inp_scale 10. --rec_scale 5. --nb_layers 2
  python main.py --model_type adLIF --dataset_name shd --nb_epochs 200 --method bptt --nb_hiddens 1024 --lr $lr --devices 0  --inp_scale 10. --rec_scale 5. --nb_layers 2
  python main.py --model_type RadLIF --dataset_name shd --nb_epochs 200 --method bptt --nb_hiddens 1024 --lr $lr --devices 0  --inp_scale 10. --rec_scale 5. --nb_layers 2
done
done
