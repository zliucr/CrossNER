
### Science Domain ###

## Directly Fine-tune
python main.py --exp_name science_directly_finetune --exp_id 1 --num_tag 35 --batch_size 16 --ckpt science_integrated/pytorch_model.bin --tgt_dm science

## Jointly Train
python main.py --exp_name science_jointly_train --exp_id 1 --num_tag 35 --batch_size 16 --conll --joint --ckpt science_integrated/pytorch_model.bin --tgt_dm science

## Pre-train then Fine-tune
python main.py --exp_name science_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --ckpt science_integrated/pytorch_model.bin --tgt_dm science

## BiLSTM-CRF
python main.py --exp_name science_bilstm_wordchar --exp_id 1 --num_tag 35 --tgt_dm science --bilstm --dropout 0.5 --lr 1e-3 --usechar --emb_dim 400

## coach word+charlevel
python main.py --exp_name science_coach_wordchar --exp_id 1 --num_tag 3 --entity_enc_hidden_dim 200 --tgt_dm science --coach --dropout 0.5 --lr 1e-4 --usechar --emb_dim 400


### Music Domain ###

## Directly Fine-tune
python main.py --exp_name music_directly_finetune --exp_id 1 --num_tag 27 --batch_size 16 --ckpt music_integrated/pytorch_model.bin --tgt_dm music

## Jointly Train
python main.py --exp_name music_jointly_train --exp_id 1 --num_tag 27 --batch_size 16 --conll --joint --ckpt music_integrated/pytorch_model.bin --tgt_dm music

## Pre-train then Fine-tune
python main.py --exp_name music_pretrain_then_finetune --exp_id 1 --num_tag 27 --batch_size 16 --conll --ckpt music_integrated/pytorch_model.bin --tgt_dm music

## BiLSTM-CRF
python main.py --exp_name music_bilstm_wordchar --exp_id 1 --num_tag 27 --tgt_dm music --bilstm --dropout 0.5 --lr 1e-3 --usechar --emb_dim 400

## coach word+charlevel
python main.py --exp_name music_coach_wordchar --exp_id 1 --num_tag 3 --entity_enc_hidden_dim 200 --tgt_dm music --coach --dropout 0.5 --lr 1e-4 --usechar --emb_dim 400


### Literature Domain ###

## Directly Fine-tune
python main.py --exp_name literature_directly_finetune --exp_id 1 --num_tag 25 --batch_size 16 --ckpt literature_integrated/pytorch_model.bin --tgt_dm literature

## Jointly Train
python main.py --exp_name literature_jointly_train --exp_id 1 --num_tag 25 --batch_size 16 --conll --joint --ckpt literature_integrated/pytorch_model.bin --tgt_dm literature

## Pre-train then Fine-tune
python main.py --exp_name literature_pretrain_then_finetune --exp_id 1 --num_tag 25 --batch_size 16 --conll --ckpt literature_integrated/pytorch_model.bin --tgt_dm literature

## BiLSTM-CRF
python main.py --exp_name literature_bilstm_wordchar --exp_id 1 --num_tag 25 --tgt_dm literature --bilstm --dropout 0.5 --lr 1e-3 --usechar --emb_dim 400

## coach word+charlevel
python main.py --exp_name literature_coach_wordchar --exp_id 1 --num_tag 3 --entity_enc_hidden_dim 200 --tgt_dm literature --coach --dropout 0.5 --lr 1e-4 --usechar --emb_dim 400


### AI Domain ###

## Directly Fine-tune
python main.py --exp_name ai_directly_finetune --exp_id 1 --num_tag 29 --batch_size 16 --ckpt ai_integrated/pytorch_model.bin --tgt_dm ai

## Jointly Train
python main.py --exp_name ai_jointly_train --exp_id 1 --num_tag 29 --batch_size 16 --conll --joint --ckpt ai_integrated/pytorch_model.bin --tgt_dm ai

## Pre-train then Fine-tune
python main.py --exp_name ai_pretrain_then_finetune --exp_id 1 --num_tag 29 --batch_size 16 --conll --ckpt ai_integrated/pytorch_model.bin --tgt_dm ai

## BiLSTM-CRF
python main.py --exp_name ai_bilstm_wordchar --exp_id 1 --num_tag 29 --tgt_dm ai --bilstm --dropout 0.5 --lr 1e-3 --usechar --emb_dim 400

## coach word+charlevel
python main.py --exp_name ai_coach_wordchar --exp_id 1 --num_tag 3 --entity_enc_hidden_dim 200 --tgt_dm ai --coach --dropout 0.5 --lr 1e-4 --usechar --emb_dim 400

