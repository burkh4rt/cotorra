# Cotorra: Configurable training

> 🦜 the wild parakeet of the south side

<img src="img/monk-parakeets-calumet-park.jpeg" alt="Monk parakeets as seen in
Calumet Park, Chicago, 12 November 2024" width="400" style="display: block;
margin: 0 auto; -webkit-mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );
  mask-image: radial-gradient(
    ellipse at center,
    rgba(0,0,0,1) 50%,
    rgba(0,0,0,0) 100%
  );"/>

## About

This repo provides a configurable trainer for generative event models on
tokenized timelines. _Cotorra_ is a Spanish term for a small-to-medium sized
parrot, particularly the Monk parakeet. Monk parakeets were introduced to the
south side of Chicago, where they have flourished. [^1]

## Installation

You can download and install this package as follows:

```sh
git clone git@github.com:bbj-lab/cotorra.git
cd cotorra
python -m venv .venv
. .venv/bin/activate
pip install "quick-sco-re @ git+https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git"
pip install -e . \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

## Context

Suppose you have a dataset of tokenized timelines `tokens_times.parquet` as a
parquet table with columns:

- `subject_id`
- `tokens` — the integer token sequence for the subject's timeline.
- `times` — a parallel list of timestamps, one per token, indicating when each
  event occurred.

The table will look something like this:

```
┌────────────────────┬─────────────────┬─────────────────────────────────┐
│ subject_id         ┆ tokens          ┆ times                           │
│ ---                ┆ ---             ┆ ---                             │
│ str                ┆ list[u32]       ┆ list[datetime[μs]]              │
╞════════════════════╪═════════════════╪═════════════════════════════════╡
│ 20002103           ┆ [20, 350, … 21] ┆ [2116-05-08 02:45:00, 2116-05-… │
│ 20008372           ┆ [20, 350, … 21] ┆ [2110-10-30 13:03:00, 2110-10-… │
│ …                  ┆ …               ┆ …                               │
│ 29994865           ┆ [20, 364, … 21] ┆ [2111-01-28 21:49:00, 2111-01-… │
└────────────────────┴─────────────────┴─────────────────────────────────┘
```

You also have a `tokenizer.yaml`, a plain yaml file that contains information
about the configuration, learned vocabulary, and bins. This file is sufficient to
reconstitute the tokenizer object. We only need this file to contain a lookup
table:

```yaml
lookup:
  UNK: 0
  ADMN//direct: 1
  ADMN//ed: 2
  ADMN//elective: 3
  AGE//age_Q0: 4
  ...
```

Finally, we need `subject_splits.parquet` which is a table listing out all
subject_id's and their corresponding split assignment (with splits: `train`,
`tuning`, and `held_out`):

```
┌────────────┬──────────┐
│ subject_id ┆ split    │
│ ---        ┆ ---      │
│ str        ┆ str      │
╞════════════╪══════════╡
│ 21081215   ┆ train    │
│ 20302177   ┆ train    │
│ …          ┆ …        │
│ 28150003   ┆ held_out │
│ 22151813   ┆ held_out │
└────────────┴──────────┘
```

<!-- prettier-ignore-start -->
> [!TIP]
> For getting your data to this point, check out our configurable
> collator / tokenizer: [☕️ cocoa](https://github.com/bbj-lab/cocoa)
<!-- prettier-ignore-end -->

Given these things, we want to train a model to predict the next token in a
subject's timeline given their complete history or context up to this point. This
package is designed to do that in a configurable way.

## Configuration

This library can be extensively customized for your use purposes through a yaml
configuration, as opposed to having to having to write python. All configuration
lives under `config/`. The entrypoint is `config/main.yaml`:

#### Main configuration ([example](config/main.yaml))

- **processed_data_home**: Path to processed data (tokenized timelines, splits,
  tokenizer config).
- **model_config**: Path to the model configuration YAML (e.g.,
  config/model/llama-32-lite.yaml). [see below]
- **max_seq_len**: Maximum sequence length for model input.
- **n_epochs**: Number of epochs (handled in the dataloader, not the trainer).
- **wandb**:
  - **project**: Weights & Biases project name for experiment tracking.
  - **run_name**: Name for the current run.
- **output_home**: Directory to save model outputs and checkpoints.
- **tokens_of_interest**: List of special tokens to upweight during training.
- **toi_weight**: Weight multiplier for tokens of interest in the loss function.
- **training_args**: Arguments passed to HuggingFace's
  [`TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)
- **tuning_args**: Arguments passed to HuggingFace's
  [`hyperparameter_search`](https://huggingface.co/docs/transformers/hpo_train?backends=Optuna)

<!-- prettier-ignore-start -->
> [!NOTE]
> You can also configure customizable loss functions here.
<!-- prettier-ignore-end -->

#### Model configuration ([example](config/model/llama-32-lite.yaml))

- **model_name**: Name or path of the model (e.g., meta-llama/Llama-3.2-1B).
- **model_args**: Model architecture parameters passed directly to HuggingFace's
  [`AutoConfig`](https://huggingface.co/docs/transformers/en/model_doc/auto)
  object

## Usage

To use a different dataset or schema, create new YAML files under
`config/collation/` and `config/tokenization/` and update the paths in
`config/main.yaml`, or pass your options directly to the `Trainer` object. The
`Trainer` class accepts `**kwargs` that are merged on top of the YAML config via
OmegaConf, so any config value can be overridden programmatically:

```python
from cotorra.trainer import Trainer

# Override config values at instantiation
trainer = Trainer(processed_data_home="~/other/data", output_home="~/other/output")
trainer.train()
```

### CLI

We provide a CLI:

```
 Usage: cotorra [OPTIONS] COMMAND [ARGS]...

 Configurable training for generative event models

╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.     │
│ --show-completion             Show completion for the current shell, to     │
│                               copy it or customize the installation.        │
│ --help                        Show this message and exit.                   │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ train             Train a model on tokenized data. For tokenization,        │
│                   consult the cocoa package.                                │
│ tune              Run hyperparameter tuning while training a model.         │
│ extract           Extract representations from a trained model.             │
│ generative-score  Generate SCORE/REACH metrics from a trained model and     │
│                   save them to parquet.                                     │
│ rep-based-score   Generate rep-based scores for the token-based outcomes of │
│                   interest.                                                 │
╰─────────────────────────────────────────────────────────────────────────────╯
```

with commands:

- `cotorra train`

  ```
  Usage: cotorra train [OPTIONS]

  Train a model on tokenized data. For tokenization, consult the cocoa package.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │ --main-config          -m      PATH  Main configuration file (overrides     │
  │                                      default)                               │
  │ --model-config                 PATH  Model configuration file               │
  │ --processed-data-home  -p      TEXT  Processed data directory (overrides    │
  │                                      config)                                │
  │ --output-home          -o      TEXT  Output directory for trained models    │
  │ --verbose              -v            Verbose logging for collate            │
  │ --help                               Show this message and exit.            │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra tune`

  ```
  Usage: cotorra tune [OPTIONS]

  Run hyperparameter tuning while training a model.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │ --main-config          -m      PATH  Main configuration file (overrides     │
  │                                      default)                               │
  │ --model-config                 PATH  Model configuration file               │
  │ --processed-data-home  -p      TEXT  Processed data directory (overrides    │
  │                                      config)                                │
  │ --output-home          -o      TEXT  Output directory for trained models    │
  │ --verbose              -v            Verbose logging for collate            │
  │ --help                               Show this message and exit.            │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra generative-score`

  ```
  Usage: cotorra generative-score [OPTIONS]

  Generate SCORE/REACH metrics from a trained model and save them to parquet.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │ --main-config          -m      PATH  Main configuration file (overrides     │
  │                                      default)                               │
  │ --processed-data-home  -p      TEXT  Processed data directory (overrides    │
  │                                      config)                                │
  │ --output-home          -o      TEXT  Output directory for score files       │
  │ --help                               Show this message and exit.            │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra extract`

  ```
  Usage: cotorra extract [OPTIONS]

  Extract representations from a trained model.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │ --main-config          -m      PATH  Main configuration file (overrides     │
  │                                      default)                               │
  │ --processed-data-home  -p      TEXT  Processed data directory (overrides    │
  │                                      config)                                │
  │ --output-home          -o      TEXT  Output directory for trained models    │
  │ --help                               Show this message and exit.            │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

- `cotorra rep-based-score`

  ```
  Usage: cotorra rep-based-score [OPTIONS]

  Generate rep-based scores for the token-based outcomes of interest.

  ╭─ Options ───────────────────────────────────────────────────────────────────╮
  │ --main-config          -m      PATH  Main configuration file (overrides     │
  │                                      default)                               │
  │ --processed-data-home  -p      TEXT  Processed data directory (overrides    │
  │                                      config)                                │
  │ --help                               Show this message and exit.            │
  ╰─────────────────────────────────────────────────────────────────────────────╯
  ```

[^1]:
    L. Gersony, "The Quiet Victory of Chicago’s Monk Parakeets," _The Chicago
    Maroon_, 23 January 2022,
    https://chicagomaroon.com/28830/grey-city/quiet-protest-chicagos-monk-parakeets/

<!--

```
tmux new -s co || tmux a -t co
```

Send to randi:
```
rsync -avht \
 --delete \
 --exclude "output/" \
 --exclude "wandb/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cotorra \
 randi:/gpfs/data/bbj-lab/users/burkh4rt
```

Send to bbj-lab1:
```
rsync -avht \
 --delete \
 --exclude "output/" \
 --exclude "wandb/" \
 --exclude ".venv/" \
 --exclude ".idea/" \
 ~/Documents/chicago/cotorra \
 bbj-lab1:~
```

-->
