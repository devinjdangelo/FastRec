python3 geosim.py

python3 SageSimEmbed.py \
--num-epochs 150 --batch-size 25 \
--test-every 1 --lr .01 --n-classes 20 \
--p-train 1 --distance-metric mse --embedding-dim 2 \
--sup-weight 1 --neg_samples 10 --agg-type gcn

python3 SageSimEmbed.py \
--num-epochs 150 --batch-size 25 \
--test-every 1 --lr .01 --n-classes 20 \
--p-train 1 --distance-metric mse --embedding-dim 2 \
--sup-weight 0 --neg_samples 1 --agg-type gcn

python3 SageSimEmbed.py \
--num-epochs 300 --batch-size 25 \
--test-every 25 --lr .01 --n-classes 40 \
--p-train 0.5 --distance-metric mse --embedding-dim 2 \
--sup-weight 1 --neg_samples 10 --agg-type gcn

python3 SageSimEmbed.py \
--num-epochs 1000 --batch-size 25 \
--test-every 25 --lr .01 --n-classes 40 \
--p-train 0.5 --distance-metric mse --embedding-dim 2 \
--sup-weight 0.5 --neg_samples 10 --agg-type gcn

python3 SageSimEmbed.py \
--num-epochs 100 --batch-size 1000 \
--test-every 10 --lr .01 --n-classes 10000 \
--p-train 0.1 --distance-metric cosine --embedding-dim 16 \
--sup-weight 0.5 --neg_samples 1 --agg-type gcn

python3 SageSimEmbed.py \
--num-epochs 100 --batch-size 1000 \
--test-every 10 --lr .01 --n-classes 20000 \
--p-train 0.05 --distance-metric cosine --embedding-dim 16 \
--sup-weight 0.5 --neg_samples 1 --agg-type gcn


python3 geosim.py --npaths 10000 --nfiles 100

python3 SageSimEmbed.py \
--num-epochs 100 --batch-size 1000 \
--test-every 10 --lr .01 --n-classes 100000 \
--p-train 0.01 --distance-metric cosine --embedding-dim 16 \
--sup-weight 0.5 --neg_samples 1 --agg-type gcn

