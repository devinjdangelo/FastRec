python3 ./sse/SageSimEmbed.py \
--num-epochs 100 --batch-size 99 \
--test-every 10 --lr .01 --n-classes 10000 \
--p-train 0.01 --distance-metric cosine --embedding-dim 512 --num-hidden 512 \
--sup-weight 0 --neg_samples 1 --agg-type gcn

python3 ./sse/SageSimEmbed.py \
--num-epochs 100 --batch-size 1000 \
--test-every 10 --lr .01 --n-classes 100000 \
--p-train 0.01 --distance-metric cosine --embedding-dim 512 --num-hidden 512 \
--sup-weight 0.5 --neg_samples 1 --agg-type gcn --device cpu --save


python3 ./sse/SageSimEmbed.py \
--num-epochs 100 --batch-size 1000 \
--test-every 10 --lr .01 --n-classes 300000 \
--p-train 0.01 --distance-metric cosine --embedding-dim 512 --num-hidden 512 \
--sup-weight 0.5 --neg_samples 1 --agg-type gcn --device cpu --load --save

python3 ./sse/SageSimEmbed.py \
--num-epochs 100 --batch-size 1000 \
--test-every 10 --lr .01 --n-classes 600000 \
--p-train 0.01 --distance-metric cosine --embedding-dim 512 --num-hidden 512 \
--sup-weight 0.5 --neg_samples 1 --agg-type gcn --device cpu --save \
--max-test-labels 100000

