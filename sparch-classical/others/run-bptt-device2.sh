

cd ..

for lr in 0.02 0.01 0.005 0.001
do
  python main.py --model_type LIF --dataset_name shd  --nb_epochs 100 --method bptt --nb_hiddens 1024 --lr $lr --devices 2
  python main.py --model_type RLIF --dataset_name shd  --nb_epochs 100 --method bptt --nb_hiddens 1024 --lr $lr --devices 2
  python main.py --model_type adLIF --dataset_name shd --nb_epochs 100 --method bptt --nb_hiddens 1024 --lr $lr --devices 2
  python main.py --model_type RadLIF --dataset_name shd --nb_epochs 100 --method bptt --nb_hiddens 1024 --lr $lr --devices 2
done
