# SuperLog & NLPLog
## NLPLog Dataset
**NLPLog Data Path**: `data/NLPLog.json`

## Continual Pre-training For SuperLog
1. Environment Configuration
```bash
pip install -e ".[torch,metrics,deepspeed]"
```
2. Start Training
```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /home/l00650981/models/Llama-2-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset NLPLog \
    --cutoff_len 2048 \
    --learning_rate 1e-05 \
    --num_train_epochs 1.5 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Llama-2-7B/full/SuperLog \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed cache/ds_z3_config.json 
```
## Instruction Following SFT For SuperLog

```bash
```
