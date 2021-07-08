NAME="HSDtransformer_ace"
SAVE="save/$NAME"

lenpen=1
beam=1
SET=test

CUDA_VISIBLE_DEVICES=1 fairseq-generate --data-dir data-bin/ace05/ --bert-path albert-xxlarge-v1  --path "$SAVE/checkpoint_last10_avg.pt"  --max-tokens 5000 --beam $beam  --lenpen $lenpen --gen-subset $SET --task seqie --max-source-positions 512 

python evaluate.py --data-dir data-bin/ace05/ --bert-path  albert-xxlarge-v1 --path "" --gen-subset $SET --task seqie --max-source-positions 512 


