# ReFine
python refine_train.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4
python refine_train.py --dataset mnist --hid 50 --epoch 50 --ratio 0.2 --lr 1e-3
python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3
python refine_train.py --dataset vg --hid 250 --epoch 100 --ratio 0.2 --lr 1e-3

# PGExplainer
python pg_train.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4
python pg_train.py --dataset mnist --hid 50 --epoch 50 --ratio 0.2 --lr 1e-3
python pg_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3
python pg_train.py --dataset vg --hid 250 --epoch 100 --ratio 0.2 --lr 1e-3

# Mutag
for re in $(seq -s ' ' 0 6); do python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3 --cuda=6 --reg_enlarge=$re; done
for re in $(seq -s ' ' 7 13); do python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3 --cuda=6 --reg_enlarge=$re; done
for re in $(seq -s ' ' 14 20); do python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3 --cuda=6 --reg_enlarge=$re; done
for re in $(seq -s ' ' 0.05 0.05 0.35); do python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3 --cuda=7 --reg_enlarge=$re; done
for re in $(seq -s ' ' 0.4 0.05 0.7); do python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3 --cuda=7 --reg_enlarge=$re; done
for re in $(seq -s ' ' 0.75 0.05 0.95); do python refine_train.py --dataset mutag --hid 100 --epoch 100 --ratio 0.4 --lr 1e-3 --cuda=7 --reg_enlarge=$re; done

# BA3
for re in $(seq -s ' ' 0.05 0.05 0.95); do python refine_train.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4 --cuda=0 --reg_enlarge=$re; done
for re in $(seq -s ' ' 1 20); do python refine_train.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4 --cuda=3 --reg_enlarge=$re; done

# MNIST
for re in $(seq -s ' ' 0.05 0.05 0.45); do python refine_train.py --dataset mnist --hid 50 --epoch 50 --ratio 0.2 --lr 1e-3 --cuda=0 --reg_enlarge=$re; done
for re in $(seq -s ' ' 0.5 0.05 0.95); do python refine_train.py --dataset mnist --hid 50 --epoch 50 --ratio 0.2 --lr 1e-3 --cuda=6 --reg_enlarge=$re; done
for re in $(seq -s ' ' 0 10); do python refine_train.py --dataset mnist --hid 50 --epoch 50 --ratio 0.2 --lr 1e-3 --cuda=7 --reg_enlarge=$re; done
for re in $(seq -s ' ' 11 20); do python refine_train.py --dataset mnist --hid 50 --epoch 50 --ratio 0.2 --lr 1e-3 --cuda=3 --reg_enlarge=$re; done