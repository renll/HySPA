# Training
export MKL_THREADING_LAYER=GNU
NAME="HSDtransformer_ace"
SAVE="save/$NAME"
N=2,3
mkdir -p $SAVE
mkdir -p tb

CUDA_VISIBLE_DEVICES=$N $(which fairseq-train) --data-dir data-bin/ace05/ \
    --task seqie \
    --arch HSDtransformer_alb --share-decoder-input-output-embed \
    --max-source-positions 512 --bert-path albert-xxlarge-v1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.25 \
    --lr '2e-4'  --lr-scheduler inverse_sqrt --warmup-updates 2000 --warmup-init-lr '1e-07'\
    --lr-slow-rate 0.1 \
    --weight-decay 0.05 \
    --update-freq 1 \
    --criterion span_ce_smooth --label-smoothing 0.1 --report-accuracy \
    --keep-last-epochs 10  --max-update 25000 --max-tokens 1024\
    --validate-after-updates 20000 \
    --tensorboard-logdir tb/$NAME \
    --save-dir $SAVE \
    --no-progress-bar \
    --num-workers 0 \
    --use-old-adam \
    --ddp-backend=no_c10d 

python scripts/average_checkpoints.py --inputs $SAVE --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

