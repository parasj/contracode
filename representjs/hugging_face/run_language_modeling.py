"""Script to run HuggingFace LM pre-training on the (augmented) code datasets."""
# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.

from dataclasses import dataclass, field
import gc
import glob
import logging
import math
import os
import pandas as pd
import pyarrow.feather as feather
from typing import Optional
from tqdm.contrib.concurrent import thread_map
import torch
from torch.utils.data.dataset import Dataset

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BertTokenizerFast,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."},
    )
    model_type: Optional[str] = field(
        default=None, metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    eval_data_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False, metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={"help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."},
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."},
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


class BERTPretokenizedPretrainingDataset(Dataset):
    def __init__(self, data_path: str, shuffle=False, train=True, max_len=500):
        super().__init__()
        logger.info("Loading data from {}".format(data_path))
        gc.disable()
        files = glob.glob(f"{data_path}*")
        logger.info("File list {}".format(", ".join(files)))
        dfs = thread_map(feather.read_feather, files, max_workers=16)
        self.data_df = pd.concat(dfs)
        gc.enable()
        logger.info(f"Loaded dataset from {data_path} with {len(self.data_df)} samples")
        self.max_len = max_len

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, i):
        return torch.tensor(self.data_df["toks"][i][: self.max_len], dtype=torch.long)


def get_dataset(
    args: DataTrainingArguments, evaluate: bool = False,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return BERTPretokenizedPretrainingDataset(file_path)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    print("Config before overwrite max_position_embeddings:", config)
    config.max_position_embeddings = 4096
    config.num_hidden_layers = 6
    config.num_attention_heads = 8
    config.hidden_size = 512
    config.intermediate_size = 2048
    print("Config after overwrite max_position_embeddings:", config)

    # if model_args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    # elif model_args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
    #         "and load it from here, using --tokenizer_name"
    #     )

    logging.info("Loading tokenizer")
    if model_args.tokenizer_name:
        tokenizer = BertTokenizerFast(model_args.tokenizer_name, clean_text=True, lowercase=False, strip_accents=True)
    else:
        raise ValueError("Specify tokenizer name")

    logging.info("Loading model")
    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    logging.info("Resizing embeddings")
    model.resize_token_embeddings(len(tokenizer))
    print(len(tokenizer.get_vocab()), len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    # Get datasets
    logging.info("Loading train dataset")
    train_dataset = get_dataset(data_args) if training_args.do_train else None
    logging.info("Loading eval dataset")
    eval_dataset = get_dataset(data_args, evaluate=True,) if training_args.do_eval else None
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer, plm_probability=data_args.plm_probability, max_span_length=data_args.max_span_length,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,)

    # Initialize our Trainer
    logging.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        logging.info("Training")
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()
