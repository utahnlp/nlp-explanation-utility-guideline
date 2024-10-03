# On Evaluating Explanation Utility for Human-AI Decision Making in NLP
Repository for the EMNLP 2024 Findings paper &ldquo;On Evaluating Explanation Utility for Human-AI Decision Making in NLP&rdquo; by Fateme Hashemi Chaleshtori, Atreya Ghosal, Alexander Gill, Purbid Bambroo, and Ana Marasović

## Environment Setup
Make a new python3.10 environment:
```
conda create python=3.10 --name expl_utility
```
Install the dependencies from the requirement file:
```
pip3 install -r requirements.txt
```

## Developing Models for Tasks
Run the following command to finetune Flan-T5-3B on ContractNLI, SciFact-Open, ILDC, and EvidenceInference-v2 datasets:
```

```

## Developing Deferral Model
Run the following command to finetune Llama-2-13B on the deferral task:
```

```
Prompts for zero-shot and few-shot learning with GPT-4 are provided in the [deferral_model](./deferral_model) directory.

## Human Study Templates
You can find Qualtrics templates for different studies under the [human_study_templates](./human_study_templates) directory. These templates can help replicate the human studies according to our guidelines.

To quickly review the format of the human studies, follow the links provided in the [human_study_templates](./human_study_templates/README.md) directory.

