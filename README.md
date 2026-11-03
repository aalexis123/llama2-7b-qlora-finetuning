# LLaMA-2 7B Fine-Tuning with QLoRA

An experimental instruction fine-tuning project using **Python, Hugging Face Transformers, TRL, PEFT/LoRA, and bitsandbytes**. The notebook prepares instruction-response data, loads the gated LLaMA-2 7B chat model in 4-bit precision, trains a LoRA adapter, and demonstrates text generation.

## Project goal

Explore parameter-efficient fine-tuning for technical question answering. The project used a **771-example instruction dataset** covering topics such as blockchain, IoT, cryptography, and distributed systems (project summary; the original CSV is not included here).

## What the notebook contains

1. Environment setup and Hugging Face authentication
2. CSV ingestion and construction of `###Human:` / `###Assistant:` examples
3. A 95% training / 5% evaluation split
4. 4-bit NF4 base-model loading and LoRA configuration
5. A custom completion-only data collator that masks prompt tokens
6. Supervised fine-tuning with `SFTTrainer`
7. Saving and reloading the adapter for inference
8. Sample text generation and optional adapter upload

## Tech stack

Python · PyTorch · Transformers · TRL · PEFT · bitsandbytes · Hugging Face Datasets · Pandas · Google Colab

## Repository files

- `llama2_finetuning_github_ready.ipynb` — annotated training and inference notebook
- `README.md` — project overview and setup notes

## How to run

1. Open the notebook in a **GPU-enabled Google Colab** runtime with enough GPU memory for 4-bit LLaMA-2 7B training.
2. Obtain approval to access [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and authenticate through the notebook's interactive Hugging Face login.
3. Upload an authorized `trainpt2.csv` file to `/content/` or update the `csv_path` variable. The CSV must contain either a `text` column or both `question` and `answer` columns.
4. Run the notebook in order. Depending on current package releases, you may need to pin compatible dependency versions.
5. Inspect training logs, saved adapter files, and generated responses. Do not run the final upload cell unless you intend to publish the adapter.

**Security:** Never commit access tokens, API keys, private datasets, or full model weights. The gated base model is not included in this repository.

## Training configuration in the notebook

| Setting | Value |
|---|---|
| Base model | `meta-llama/Llama-2-7b-chat-hf` |
| Quantization | 4-bit NF4 |
| LoRA rank / alpha | 64 / 16 |
| LoRA dropout | 0.1 |
| Epochs | 1 |
| Batch size / gradient accumulation | 1 / 8 |
| Learning rate | `2e-4` |
| Maximum sequence length | 1024 |
| Evaluation split | 5% |

## Results and limitations

The notebook contains code for training, saving, reloading, and qualitative prompt testing. **It does not provide a verified held-out accuracy score or a reproducible evaluation report.** Generated examples are demonstrations, not evidence of reliable domain expertise. The configuration's evaluation interval may be longer than the run, so inspect actual logs before claiming validation results.

The model adapter was associated with the Hugging Face repository [`Alexis987/llama2-7b-lora`](https://huggingface.co/Alexis987/llama2-7b-lora); check that page for current availability and access conditions.

## Next improvements

- Publish a sanitized sample dataset and its schema.
- Pin tested dependency versions and record runtime hardware.
- Add a held-out evaluation set with reference answers and reproducible metrics.
- Compare base-model and adapter responses on identical prompts.
- Add a lightweight inference script and automated tests.

## Author

Alexis Amadi — Computer Information Systems graduate focused on data engineering and applied AI.
