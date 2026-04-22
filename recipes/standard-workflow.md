## Typical workflow

1. We start with [MIMIC data](https://mimic.mit.edu) that's been converted to the
   [CLIF standard](https://clif-icu.com):
   https://physionet.org/content/mimic-iv-ext-clif. We first collate and tokenize
   it with the [cocoa package](https://github.com/bbj-lab/cocoa).

<details>

<summary>Localize filenames by cluster.</summary>

```sh
case "$(uname -n)" in
    cri*)
        hm="/gpfs/data/bbj-lab/users/burkh4rt"
        ;;
    bbj-lab*)
        hm="/mnt/bbj-lab/users/burkh4rt"
        ;;
    *)
        echo "Unknown host $(uname -n)"
        ;;
esac
```

</details>

```sh
cocoa pipeline \
    --raw-data-home "${hm}/development-sample-21/raw-mimic/dev" \
    --processed-data-home ./processed/dev \
    --verbose
```

2. Next we train a model on this data (with hyperparameter tuning):

```sh
cotorra tune \
    --processed-data-home ../cocoa/processed/dev \
    --output-home ./output/dev/ \
    --model-config ./config/model/llama-32-lite.yaml \
    --verbose
```

3. You can then extract reps with:

```sh
cotorra extract \
    --processed-data-home ../cocoa/processed/dev \
    --output-home ./output/dev/
```

4. You can get generative predictions with:

```sh
cotorra score \
    --processed-data-home ../cocoa/processed/dev \
    --output-home ./output/dev/
```

_However, [sglang](https://docs.sglang.io) needs to be setup in order for this
last step to run._
