

#
#python dvs128.py --data ./data/ --cache ./cache/ \
#    --logdir ./logs/ --batch-size 40 --units 795 --unit-size 1 \
#    --num-layers 1 --frame-size 128 --run-title egrud795_rerun \
#    --train-epochs 500 --frame-time 25 --rnn-type egru \
#    --learning-rate 0.0009975 --lr-gamma 0.8747 --lr-decay-epochs 56 \
#    --event-agg-method mean --use-cnn --use-all-timesteps \
#    --dropout 0.6321 --dropconnect 0.08134 --zoneout 0.2319 \
#    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
#    --activity-regularization --activity-regularization-constant 0.01 \
#    --augment-data
#
#
#
#
#python dvs128.py --data ./data/ --cache ./cache/ \
#    --logdir ./logs/ --batch-size 40 --units 795 --unit-size 1 \
#    --num-layers 1 --frame-size 128 --run-title egrud795_rerun \
#    --train-epochs 500 --frame-time 25 --rnn-type egru \
#    --learning-rate 0.0009975 --lr-gamma 0.8747 --lr-decay-epochs 56 \
#    --event-agg-method mean  --use-all-timesteps \
#    --dropout 0.6321 --dropconnect 0.08134 --zoneout 0.2319 \
#    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
#    --activity-regularization --activity-regularization-constant 0.01 \
#    --augment-data
#
#
#
#
#
#python train_shd_ssc.py --data D:/codes/githubs/SNN/EvNN/data/ --cache D:/codes/githubs/SNN/EvNN/cache/  --batch-size 40 --units 795 --unit-size 1 \
#    --num-layers 1 --frame-size 128 --run-title egrud795_rerun \
#    --train-epochs 500 --frame-time 25 --rnn-type gru \
#    --learning-rate 0.0009975 --lr-gamma 0.8747 --lr-decay-epochs 56 \
#    --event-agg-method mean --use-cnn  --use-all-timesteps \
#    --dropout 0.6321 --dropconnect 0.08134 --zoneout 0.2319 \
#    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
#    --activity-regularization --activity-regularization-constant 0.01 \
#    --augment-data
#
#
#



#
#
#python train_shd_ssc.py --batch-size 40 --units 795 --unit-size 1 \
#    --num-layers 1 --frame-size 128 --run-title egrud795_rerun \
#    --train-epochs 500 --frame-time 25 --rnn-type gru \
#    --learning-rate 0.0009975 --lr-gamma 0.8747 --lr-decay-epochs 56 \
#    --event-agg-method mean --use-cnn  --use-all-timesteps \
#    --dropout 0.6321 --dropconnect 0.08134 --zoneout 0.2319 \
#    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
#    --activity-regularization --activity-regularization-constant 0.01 \
#    --augment-data


# Event-GRU, BPTT
python bst_main.py --batch-size 40 --units 795 --unit-size 1 \
    --num-layers 1 --frame-size 128 --method bptt  \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 56 \
    --event-agg-method mean --use-cnn  --use-all-timesteps \
    --dropout 0.5 --dropconnect 0.08134 --zoneout 0.2319 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
    --activity-regularization --activity-regularization-constant 0.01 \
    --augment-data  --devices 0


# Event-GRU, D-RTRL
python bst_main.py --batch-size 40 --units 795 --unit-size 1 \
    --num-layers 1 --frame-size 128 --method d-rtrl \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 56 \
    --event-agg-method mean --use-cnn  --use-all-timesteps \
    --dropout 0.5 --dropconnect 0.08134 --zoneout 0.2319 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
    --activity-regularization --activity-regularization-constant 0.01 \
    --augment-data  --devices 0



# GRU, BPTT
python bst_main.py --batch-size 40 --units 795 --unit-size 1 \
    --num-layers 1 --frame-size 128 --method bptt  \
    --train-epochs 500 --frame-time 25 --rnn-type gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 56 \
    --event-agg-method mean --use-cnn  --use-all-timesteps \
    --dropout 0.5 --dropconnect 0.08134 --zoneout 0.2319 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
    --activity-regularization --activity-regularization-constant 0.01 \
    --augment-data  --devices 0


# LSTM, D-RTRL
python bst_main.py --batch-size 40 --units 795 --unit-size 1 \
    --num-layers 1 --frame-size 128 --method d-rtrl \
    --train-epochs 500 --frame-time 25 --rnn-type gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 56 \
    --event-agg-method mean --use-cnn  --use-all-timesteps \
    --dropout 0.5 --dropconnect 0.08134 --zoneout 0.2319 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
    --activity-regularization --activity-regularization-constant 0.01 \
    --augment-data  --devices 0



# LSTM, BPTT
python bst_main.py --batch-size 40 --units 795 --unit-size 1 \
    --num-layers 1 --frame-size 128 --method bptt  \
    --train-epochs 500 --frame-time 25 --rnn-type lstm \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 56 \
    --event-agg-method mean --use-cnn  --use-all-timesteps \
    --dropout 0.5 --dropconnect 0.08134 --zoneout 0.2319 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
    --activity-regularization --activity-regularization-constant 0.01 \
    --augment-data  --devices 0


# GRU, D-RTRL
python bst_main.py --batch-size 40 --units 795 --unit-size 1 \
    --num-layers 1 --frame-size 128 --method d-rtrl \
    --train-epochs 500 --frame-time 25 --rnn-type lstm \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 56 \
    --event-agg-method mean --use-cnn  --use-all-timesteps \
    --dropout 0.5 --dropconnect 0.08134 --zoneout 0.2319 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.2465 \
    --activity-regularization --activity-regularization-constant 0.01 \
    --augment-data  --devices 0





