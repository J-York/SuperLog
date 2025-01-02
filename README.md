# SuperLog & NLPLog

This repository includes the training codes for LogCraft, as well as a specifically designed interpretable dataset for the log analysis domain, named NLPLog.

## NLPLog Dataset

NLPLog is a question-answer dataset generated through interactions with ChatGPT. We utilized the publicly available log datasets provided by LogHub, posing questions to ChatGPT across five defined dimensions. This approach yielded conversational data rich in log-related domain information and high interpretability. NLPLog will serve as the dataset for the subsequent continuous pre-training of SuperLog.

**NLPLog Data Path**: `data/NLPLog.json`

## Continual Pre-training For SuperLog

At this stage, we leverage the open-source LLM training framework, LLaMa-Factory, to continuously pre-train an open-source large model using the NLPLog dataset. In this step, we infuse interpretable knowledge, equipping the general-purpose LLM with domain-specific information related to log analysis.

1. Environment Configuration
```bash
pip install -e ".[torch,metrics,deepspeed]"
```
2. Start Training
```bash
llamafactory-cli train examples/train_full/train_superlog.yaml
```
## Instruction Following SFT For SuperLog

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/j50043562/temp/SuperLog/saves/Llama-2-7B/full/SuperLog \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset alpaca_1k \
    --cutoff_len 2048 \
    --learning_rate 1e-05 \
    --num_train_epochs 3.0 \
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
    --output_dir saves/Llama-2-7B/full/SuperLog_IF \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed cache/ds_z3_config.json 
```
