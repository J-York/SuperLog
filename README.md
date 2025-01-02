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

In this step, we fine-tune SuperLog using a selection of 1000 high-quality instruction data from Alpacar. The purpose of this step is to enable SuperLog to have a good ability to respond to human instructions, understand log analysis commands, and complete the corresponding tasks.

```bash
llamafactory-cli train examples/train_full/train_superlog.yaml
```
