python dummy_data_prep.py

python train.py \
    --tokens dummy_data/tokens.npy \
    --vae-embeddings dummy_data/vae_embeddings.npy \
    --vocab-size 256 \
    --latent-dim 128 \
    --d-model 512 \
    --nhead 8 \
    --num-decoder-layers 6 \
    --dim-feedforward 2048 \
    --dropout 0.1 \
    --max-seq-len 512 \
    --num-latent-tokens 4 \
    --batch-size 8 \
    --epochs 10 \
    --lr 1e-4 \
    --save-path music_transformer.pt \
    --seq-len 128

python inference_example.py
