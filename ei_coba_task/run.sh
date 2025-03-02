# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-


for i in {1..10}
do
  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method bptt  --n_rec 400 --neuron gifv3

  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method d-rtrl  --n_rec 400 --neuron gifv3 --epoch_per_step 10
  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method d-rtrl  --n_rec 400 --neuron gifv3 --epoch_per_step 5 --mode sim

  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method esd-rtrl --etrace_decay 0.95  --n_rec 400 --neuron gifv3
  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method esd-rtrl --etrace_decay 0.98  --n_rec 400 --neuron gifv3
  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method esd-rtrl --etrace_decay 0.99  --n_rec 400 --neuron gifv3
done


#
#for i in {1..10}
#do
#  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method bptt  --n_rec 400 --diff_spike 1 --neuron gifv3
#
#  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method d-rtrl  --n_rec 400
#
#  python training.py --tau_neu 400 --tau_syn 5 --tau_a 1500  --ff_scale 1.0 --rec_scale 0.5 --method esd-rtrl --etrace_decay 0.98  --n_rec 400
#done

