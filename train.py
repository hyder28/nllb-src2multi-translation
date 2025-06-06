#!/usr/bin/env python
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi Lingual Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
import jsonlines

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

flores_lang_code_mapping ={
    "en-US": "eng_Latn",
    "zh-CN": "zho_Hans",
    "es-ES": "spa_Latn",
    "fr-FR": "fra_Latn",
    "de-DE": "deu_Latn",
    "it-IT": "ita_Latn",
    "ja-JP": "jpn_Jpan",
    "ko-KR": "kor_Hang",
    "pt-BR": "por_Latn",
    "ru-RU": "rus_Cyrl",
    "ar-SA": "ara_Arab",
    "hi-IN": "hin_Deva",
}

def concat_tensor_to_len(t, dim, max_len, value):
    t_shape = list(t.shape)
    t_shape[dim] = max_len - t.shape[dim]
    assert t_shape[dim] >= 0
    pad = torch.full(t_shape, value)
    return torch.concat([t, pad], dim = dim)


def get_langs(examples, source_lang):
    tgt_langs = []
    for ex in examples:
        lang = [lang for lang in ex if lang not in [source_lang] and ex[lang] != ""]
        if len(lang) != 1:
            logger.info("ex: {}".format(ex))
            logger.error("lang: {}".format(lang))
            raise ValueError("One example must have on language pair only")
        lang = lang[0]
        tgt_langs.append(lang)
    return tgt_langs

class MultilingualSeq2SeqTrainer(Seq2SeqTrainer):
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = "eval", **gen_kwargs):
        eval_dataset = self.eval_dataset
        eval_results = {}
        for lang_pair, dataset in eval_dataset.items():
            source_lang, target_lang = lang_pair.split("#")
            global current_stage, current_target_lang
            current_stage = "validation"
            current_target_lang = target_lang
            logger.info("*" * 60)
            logger.info("Running evaluation for lang pair {}".format(lang_pair))
            logger.info("*" * 60)

            # Prepare forced_bos_token_id for this specific langauge evaluation
            forced_bos_token_id = None
            if hasattr(self.tokenizer, "lang_code_to_id"):
                forced_bos_token_id = self.tokenizer.lang_code_to_id(flores_lang_code_mapping[target_lang])

            # Perform evaluation with forced_bos_token_id as a generation argument
            result = super().evaluate(eval_dataset = dataset, ignore_keys = ignore_keys, metric_key_prefix = metric_key_prefix, forced_bos_token_id = forced_bos_token_id, **gen_kwargs)
            result["eval_samples"] = len(dataset)

            eval_results[lang_pair] = result

            self.log_metrics("eval_{}_ep{}".format(lang_pair, self.state.epoch), result)
            self.save_metrics("eval_{}_ep{}".format(lang_pair, self.state.epoch), result)

        return eval_results
    
    def predict(self, test_dataset, ignore_keys = None, metric_key_prefix = "test", **gen_kwargs):
        """Overrides predict method to handle a dictionary of datasets"""
        if isinstance(test_dataset, dict):
            predictions = {}
            for lang_pair, dataset in test_dataset.items():
                source_lang, target_lang = lang_pair.split("#")
                global current_stage, current_target_lang
                current_stage = "predict"
                current_target_lang = target_lang

                logger.info("*" * 60)
                logger.info("Running prediction for lang pair {}".format(lang_pair))
                logger.info("*" * 60)

                forced_bos_token_id = None
                if hasattr(self.tokenizer, "lang_code_to_id"):
                    forced_bos_token_id = self.tokenizer.lang_code_to_id(flores_lang_code_mapping[target_lang])

                result = super().predict(test_dataset = dataset, ignore_keys = ignore_keys, metric_key_prefix = f"{metric_key_prefix}_{lang_pair}", forced_bos_token_id = forced_bos_token_id, **gen_kwargs)
                predictions[lang_pair] = result
            
            return predictions
        
        # if not a dict, fallback to the normal predict method
        return super().predict(test_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
                

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "A list of target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )

    glossary_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input glossary data file for better term translation and computing related metrics."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    term_augmentation: bool = field(
        default=None,
        metadata={
            "help": "Method to augment terms: none, sample"
        },
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")
        
        self.target_lang = self.target_lang.split(",")
        self.validation_file = self.validation_file.split(",") if self.validation_file is not None else None
        self.test_file = self.test_file.split(",") if self.test_file is not None else None
        self.glossary_file = self.glossary_file.split(",") if self.glossary_file is not None else None

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the language codes for input/target.
    source_lang = data_args.source_lang
    target_lang = data_args.target_lang
    logger.info(f"source_lang: {}, target_lang: {}".format(source_lang, target_lang)) # source_lang: en-US, target_lang: ['zh-CN', 'ru-RU']


    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            if len(data_args.validation_file) != len(target_lang):
                logger.info("target_lang: {}".format(target_lang))
                logger.info("validation_file: {}".format(data_args.validation_file))
                raise ValueError("validation_file must match target lang")
            for lang, fname in zip(target_lang, data_args.validation_file):
                data_files["validation.{}.{}".format(source_lang.replace("-", "_"), lang.replace("-", "_"))] = fname
            extension = data_args.validation_file[0].split(".")[-1]
        if data_args.test_file is not None:
            if len(data_args.test_file) != len(target_lang):
                logger.info("target_lang: {}".format(target_lang))
                logger.info("test_file: {}".format(data_args.test_file))
                raise ValueError("test_file must match target lang")
            for lang, fname in zip(target_lang, data_args.test_file):
                data_files["test.{}.{}".format(source_lang.replace("-", "_"), lang.replace("-", "_"))] = fname
            extension = data_args.test_file[0].split(".")[-1]

        if extension == "jsonl":
            builder_name = "json"
        else:
            builder_name = extension

        raw_datasets = load_dataset(
            builder_name,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Load glossary for training and computing EMTA metric
    if data_args.glossary_file is not None:
        if len(data_args.glossary_file) != len(target_lang):
            logger.info("target_lang: {}".format(target_lang))
            logger.info("glossary_file: {}".format(data_args.glossary_file))
            raise ValueError("glossary_file must match target lang")
        glossary = {}
        glossary_lowersrc = {}
        src_terms = {}
        tgt_terms = {}
        for lang, fname in zip(target_lang, data_args.glossary_file):
            glossary[lang] = {}
            with jsonlines.open(fname, "r") as reader:
                for l in reader:
                    glossary[lang][l['translation'][source_lang]] = l['translation'][lang]
            glossary_lowersrc[lang] = {k.lower(): v for k, v in glossary[lang].items()}
            src_terms[lang] = list(glossary[lang].keys())
            tgt_terms[lang] = list(glossary[lang].values())
            logger.info("Successfully loaded glossary for {} with {} pairs".format(lang, len(glossary[lang])))

    
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                        cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir,
                                              use_fast=model_args.use_fast_tokenizer, 
                                              src_lang = flores_lang_code_mapping[source_lang])
    
    
    if len(target_lang) > 1:
        tokenizer_by_lang = {}
        for lang in target_lang:
            tokenizer_by_lang[lang] = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                                                   cache_dir=model_args.cache_dir,
                                                                   use_fast=model_args.use_fast_tokenizer,
                                                                   src_lang = flores_lang_code_mapping[source_lang],
                                                                   tgt_lang = flores_lang_code_mapping[lang])
    
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        validation_splits = []
        for lang in target_lang:
            validation_splits.append(f"validation.{source_lang.replace('-', '_')}.{lang.replace('-', '_')}")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        test_splits = []
        for lang in target_lang:
            test_splits.append(f"test.{source_lang.replace('-', '_')}.{lang.replace('-', '_')}")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    # Check the whether the source target length fits in the model, if it has absolute positional embeddings
    if (
        hasattr(model.config, "max_position_embeddings")
        and not hasattr(model.config, "relative_attention_max_distance")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        raise ValueError(
            f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
            f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
            f" `--max_source_length` to {model.config.max_position_embeddings} or using a model with larger position "
            "embeddings"
        )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples, split):
        logger.info(f"Preprocessing {} split for {} to {}".format(split, source_lang, target_lang))
        tgt_langs = get_langs(examples['transaltion'], source_lang)

        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex, tgt_lang in zip(examples["translation"], tgt_langs)]

        inputs = [prefix + inp for inp in inputs]

        if random.random() < 0.2:
            logger.info("inputs[-1]")
            logger.info(inputs[-1])
            logger.info("targets[-1]")
            logger.info(targets[-1])

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        if len(target_lang) > 1:
            targets_by_lang = defaultdict(list)
            for lang, ex in zip(tgt_langs, targets):
                targets_by_lang[lang].append(ex)
            
            # tokenize by lang
            labels_by_lang = {}
            for lang, targets in targets_by_lang.items():
                labels_by_lang[lang] = tokenizer_by_lang[lang][text_target = targets, max_length = max_target_length, padding = padding, truncation = True]
            

            # combine target tokens to one dict
            combined_input_ids, combined_attention_mask = [], []
            lang_counter = defaultdict(int)
            for lang in tgt_langs:
                lang_idx = lang_counter[lang]
                combined_input_ids.append(labels_by_lang[lang]["input_ids"][lang_idx])
                combined_attention_mask.append(labels_by_lang[lang]["attention_mask"][lang_idx])
                lang_counter[lang] += 1
            
            labels = {
                "input_ids": combined_input_ids,
                "attention_mask": combined_attention_mask
            }
        else:
            labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

            

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        if data_args.term_augmentation_sample_repeats > 0:
            # append glossary pairs to the end of training samples
            for i in range(data_args.term_augmentation_sample_repeats):
                for lang in glossary:
                    for src_term, tgt_term in glossary[lang].items():
                        train_dataset = train_dataset.add_items({"translation": {source_lang: src_term, lang: tgt_term}})

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                fn_kwargs = {"split": "train"},
                desc="Running tokenizer on train dataset",
            )
    
    val_sources = {}
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if len(data_args.validation_file) == 0:
            raise ValueError("--do_eval requires a validation dataset")

        eval_datasets = {}
        for split, eval_dataset in raw_datasets.items():
            if "validation" not in split:
                continue

            _, src_lang, tgt_lang = split.splut(".")
            src_lang = src_lang.replace("_", "-")
            tgt_lang = tgt_lang.replace("_", "-")

            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs = {"split": "val"},
                    desc = "Running tokenizer on validation dataset")
            
            lang_pair = "{}#{}".format(src_lang, tgt_lang)
            eval_datasets[lang_pair] = eval_dataset
            logger.info("*" * 60)
            logger.info("validation, lang pair: {}, number of examples: {}".format(lang_pair, len(eval_dataset)))

    predict_sources = {}
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if len(data_args.test_file) == 0:
            raise ValueError("--do_predict requires a test dataset")

        predict_datasets = {}
        for split, predict_dataset in raw_datasets.items():
            if "test" not in split:
                continue
            _, src_lang, tgt_lang = split.split(".")
            src_lang = src_lang.replace("_", "-")
            tgt_lang = tgt_lang.replace("_", "-")

            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    fn_kwargs = {"split": "test"},
                    desc = "Running tokenizer on prediction dataset")
            
            lang_pair = "{}#{}".format(src_lang, tgt_lang)
            predict_datasets[lang_pair] = predict_dataset
            logger.info("*" * 60)
            logger.info("prediction, lang pair: {}, number of examples: {}".format(lang_pair, len(predict_dataset)))

    # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
                
    # Metric
    logger.info("model_args.cache_dir: {}".format(model_args.cache_dir))

    bleu_metric = evaluate.load("sacrebleu")
    ter_metric = evaluate.load("ter")
    comet_metric = evaluate.load("comet")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, preds, tokenizer.pad_token_id)
        decoded_labels  = tokenizer.batch_decode(labels, skip_special_tokens=True)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_inputs = tokenizer.batch_decode(inputs, use_source_tokenizer = True, skip_special_tokens=True)

        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        ter_result = ter_metric.compute(predictions=decoded_preds, references=decoded_labels, ignore_punct = True)
        comet_result = comet_metric.compute(predictions = decoded_preds, references = [s[0] for s in decoded_labels], sources = decoded_inputs)

        result = {
            "bleu": bleu_result["score"],
            "ter": ter_result["score"],
            "comet": comet_result["mean_score"]
        }

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in results.items()}
        return result
    
    trainer = MultilingualSeq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset if training_args.do_train else None,
        eval_dataset = eval_datasets if training_args.do_eval else None,
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )
    current_stage = None
    current_target_lang = None

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_sample"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    if training_args.do_predict:
        logger.info("*** Predict ***")
        current_stage = "predict"
        predict_results = trainer.predict(predict_datasets, metric_key_prefix = "predict", max_length = max_length, num_beams = num_beams)

        for lang_pair, predict_result in predict_results.items():
            src_lang, tgt_lang = lang_pair.split("#")
            metrics = predict_result.metrics
            metrics["predict_samples"] = len(predict_datasets[lang_pair])

            trainer.log_metrics("predict_{}".format(lang_pair), metrics)
            trainer.save_metrics("predict_{}".format(lang_pair), metrics)

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = predict_result.predictions
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    predictions = tokenizer.batch_decode(
                        predictions, skip_special_tokens = True, clean_up_tokenization_spaces = True   
                    )
                    predictions = [pred.strip() for pred in predictions]

                    ouptut_prediction_file = os.path.join(training_args.output_dir, "generated_predictions_{}.txt".format(lang_pair))
                    with open(output_prediction_file, "w", encoding = "utf-8") as writer:
                        writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang] + data_args.target_lang if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results
            
        
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()