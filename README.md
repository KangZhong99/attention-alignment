# attention-alignment
Code for Paper "A Detection-based Attention Alignment Method for Document-level Neural Machine Translation"

### Requirements
```
3.6 <= Python <=3.8
Pytorch >= 1.4
tensorboardX, numpy
```

### Train
We implement the described models based on the [G-Transformer](https://github.com/baoguangsheng/g-transformer) for training and evaluation.
Please follow the tutorial of G-Transformer to first preprocess the data, and then train a sentence-level Transformer. For finetuning baded on the sent Transformer, please follow this script:
```
#example of training on dataset News
user_dir=./code/
data_path=/path/to/binaried_data
sent_path=/path/to/pretrained_sent_Transformer
model_path=/path/to/save/DocModel

mkdir -p $model_path
python train.py $data_path --save-dir $model_path --tensorboard-logdir $model_path \
         --user-dir $user_dir --seed 444 --num-workers 4 \
         --task my_translation_task --source-lang en --target-lang de --langs en,de \
         --arch aligned_transformer --doc-mode partial --share-all-embeddings \
         --optimizer myadam --adam-betas "(0.9, 0.998)" \
         --lr-scheduler inverse_sqrt --lr 5e-04 --warmup-updates 4000 \
         --criterion myloss --label-smoothing 0.1 --no-epoch-checkpoints \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 10 \
         --restore-file $sent_model --reset-optimizer --reset-meters --reset-dataloader 
         --reset-lr-scheduler --dropout 0.4 --flag-loss-weight 0.15 --tau 10000.0\
         --load-partial --doc-double-lr --lr-scale-pretrained 0.2 \
         --encoder-ctxlayers 2 --decoder-ctxlayers 2 --cross-ctxlayers 2 \
         --doc-noise-mask 0.1 --doc-noise-epochs 40
```
### Inference
```
user_dir=./code/
data_path=/path/to/binaried_data
model_path=/path/to/saved_DocModel
output_path=/path/to/output_result

mkdir -p $output_path
python -m fairseq_cli.generate $data_path --user-dir $user_dir --path $model_path \
         --gen-subset test --batch-size 16 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang en --target-lang de --langs en,de \
         --doc-mode partial --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $output_path
```

