# Cotorra: Configurable training

> the feral parakeet of the south side

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

You can use [uv](https://docs.astral.sh/uv/pip/) to create an environment for
running this code (with Python >= 3.12) as follows:

```sh
uv sync
uv run cotorra --help
```

[^1] L. Gersony, "The Quiet Victory of Chicago’s Monk Parakeets," _The Chicago
Maroon_, 23 January 2022,
https://chicagomaroon.com/28830/grey-city/quiet-protest-chicagos-monk-parakeets/

<!--

```
systemd-run --scope --user tmux new -s co || tmux a -t co
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

Send to bbj-lab2:
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
