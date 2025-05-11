# NLLB Model Training

This repository contains code for fine-tuning the NLLB (No Language Left Behind) model for multilingual translation tasks. The implementation supports training on multiple language pairs simultaneously and includes comprehensive evaluation metrics.

## Features

- Multilingual training support for multiple language pairs with model checkpointing
- Comprehensive evaluation metrics (BLEU, TER, COMET)
- Flexible input/output length handling
- Support for multiple target languages

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

The training data should be in JSONL format


## Supported Languages

The model supports the following languages with their corresponding FLORES codes:

| Language | FLORES Code |
|----------|-------------|
| English  | eng_Latn    |
| Chinese  | zho_Hans    |
| Spanish  | spa_Latn    |
| French   | fra_Latn    |
| German   | deu_Latn    |
| Italian  | ita_Latn    |
| Japanese | jpn_Jpan    |
| Korean   | kor_Hang    |
| Portuguese| por_Latn   |
| Russian  | rus_Cyrl    |
| Arabic   | ara_Arab    |
| Hindi    | hin_Deva    |

## Training

### Basic Training

```bash
python train.py \
    --model_name_or_path facebook/nllb-200-distilled-600M \
    --source_lang en-US \
    --target_lang zh-CN,ru-RU \
    --train_file './data/train/en-US#zh-CN#ru-RU_processed_2langs.jsonl' \
    --validation_file './data/val/en-US#zh-CN_processed_2langs.jsonl,./data/val/en-US#ru-RU_processed_2langs.jsonl' \
    --test_file './data/test/en-US#zh-CN_processed_2langs.jsonl,./data/test/en-US#ru-RU_processed_2langs.jsonl' \
    --output_dir './output/en-US#zh-CN#ru-RU' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_source_length 128 \
    --max_target_length 128 \
    --learning_rate 1e-5 \
    --fp16 \
    --save_strategy_epoch \
    --save_total_limit 2 \
    --include_inputs_for_metrics \
    --log_level info \
    --log_level_replica info
```

## Evaluation

The model is evaluated using multiple metrics:

- BLEU (Bilingual Evaluation Understudy)
- TER (Translation Edit Rate)
- COMET (Crosslingual Optimized Metric for Evaluation of Translation)

## Model Checkpoints

The training script automatically saves checkpoints in the specified output directory. You can resume training from a checkpoint using:

### Adding New Languages

To add new languages:

1. Add the language code to `flores_lang_code_mapping`
2. Update the tokenizer configuration
3. Add the language to the target languages list

### Custom Evaluation Metrics

To add custom evaluation metrics:

1. Implement the metric in the `compute_metrics` function
2. Add the metric to the evaluation pipeline

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable mixed precision training with `--fp16`
3. **Poor Performance**: Adjust learning rate or increase training data

### Best Practices

1. Use validation set for early stopping
2. Monitor training metrics regularly
3. Save checkpoints frequently
4. Use appropriate batch sizes for your hardware

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Facebook AI Research for the NLLB model
- Hugging Face for the Transformers library
- The open-source community for various tools and libraries

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{nllb2022,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={NLLB Team},
  year={2022}
}
``` 