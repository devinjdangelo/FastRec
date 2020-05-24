python3 geosim.py

python3 SageSimEmbed.py \
--num-epochs 300 --batch-size 20 \
--test-every 1 --lr .01 --n-classes 20 \
--p-train 1 --distance-metric mse --embedding-dim 2

python3 SageSimEmbed.py \
--num-epochs 400 --batch-size 1000 \
--test-every 25 --lr .01 --n-classes 10000 \
--p-train 1 --distance-metric cosine --embedding-dim 32

python3 SageSimEmbed.py \
--num-epochs 400 --batch-size 1000 \
--test-every 25 --lr .01 --n-classes 10000 \
--p-train 0.5 --distance-metric cosine --embedding-dim 32