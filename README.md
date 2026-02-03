# cs244c-zero

we’re reimplementing ZeRO stages 1–3 in pytorch and measuring the memory x communication x throughput tradeoff. the goal isn’t to clone deepspeed

## what we’re doing

- build a small decoder-only transformer that’s easy to scale.
- implement ZeRO stages (listed below).
- stage 0: classic ddp-style all-reduce on grads.
- stage 1: shard optimizer state.
- stage 2: shard grads via reduce-scatter.
- stage 3: shard params and all-gather them on the fly.
- log peak gpu memory, step time, and comm volume.
- sweep “bandwidth” with a simple sleep-based throttle to find where ZeRO-3 loses on throughput.

## repo layout

- `src/train.py`: training loop + stage logic + logging
- `src/model.py`: tiny transformer
- `src/zero/`: sharding + comm helpers
- `infra/gcp/`: one-shot gpu vm runner (terraform)
- `infra/lambda/`: one-shot gpu vm runner (terraform + lambda cli)
- `scripts/gcp_setup.sh`: installs gcloud and sets your project
- `scripts/lambda_run.sh`: runs a one-shot lambda job

## quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

sanity check on cpu (single process):

```bash
python3 src/train.py --stage 0 --steps 2 --d-model 128 --n-layers 2 --n-heads 4 --dim-ff 512 --seq-len 64 --batch-size 2
```

multi-gpu (stage 0 baseline):

```bash
torchrun --nproc_per_node=2 src/train.py --stage 0
```

stage 1 (optimizer state sharding):

```bash
torchrun --nproc_per_node=2 src/train.py --stage 1
```

stage 2 (gradient sharding via reduce-scatter):

```bash
torchrun --nproc_per_node=2 src/train.py --stage 2
```

stage 3 (parameter sharding + on-the-fly all-gather):

```bash
torchrun --nproc_per_node=2 src/train.py --stage 3
```

## useful flags

- `--bandwidth-gbps 25` to fake a slower link for collectives
- `--steps 100` to shorten runs
- `--d-model`, `--n-layers`, `--n-heads`, `--dim-ff` to scale the model
- `--batch-size`, `--seq-len` to change workload
- `--data text --data-path path/to/file.txt` to train on real text
- `--data-chars` to use character vocab instead of raw bytes

## how the stages map to code

- stage 0: all-reduce grads, standard adamw
- stage 1: all-reduce grads, sharded adamw state
- stage 2: reduce-scatter grads, sharded adamw state
- stage 3: sharded params for linear layers, all-gather on forward/backward

stage 3 details:
- only linear layers are sharded (`ZeroLinear`)
- embeddings and layernorm params stay replicated to keep it simple
- we re-gather weights in backward instead of storing full weights, which keeps peak memory lower

## comm + memory logging

each step logs:
- step time
- peak gpu memory (allocated + reserved)
- bytes communicated per collective

comm volume is an estimate based on tensor sizes, which is good enough for comparing stages.

## important constraints

- for stages 1–2 we shard params along dim 0
- that means params with dim0 not divisible by world size stay replicated
- defaults are chosen to divide cleanly for 2/4/8 gpus

## real dataset mode

by default we train on random tokens, so the loss stays flat (around log(vocab)).
if you want to see real learning, pass a text file:

```bash
python3 src/train.py --stage 0 --data text --data-path data/my_corpus.txt --seq-len 128 --batch-size 8
```

this uses a simple next-token task over bytes by default (vocab size 256). if you want a char-level vocab instead:

```bash
python3 src/train.py --stage 0 --data text --data-path data/my_corpus.txt --data-chars
```

## running in the cloud (gcp)

we use a one-shot gpu vm that runs a command and shuts down.

1) run the setup helper:

```bash
./scripts/gcp_setup.sh --project YOUR_PROJECT_ID --zone us-central1-a
```

2) edit `infra/gcp/terraform.tfvars` and set `repo_url`

3) launch:

```bash
cd infra/gcp
terraform init
terraform apply
```

4) destroy when done:

```bash
terraform destroy
```

the vm auto-shuts down after `auto_shutdown_minutes`, but you should still `terraform destroy` to clean up disks.

## running in the cloud (lambda + terraform)

this uses terraform as a wrapper around the lambda cloud cli, so we can keep a one-command workflow without gcp quotas.

fresh user checklist:

1) make a lambda cloud account and add an ssh key in the console.
2) grab your api token from the console.
3) install terraform (one time).

then from this repo:

```bash
export LAMBDA_CLOUD_API_TOKEN="your_token_here"
cd infra/lambda
terraform init
terraform apply -var "ssh_key_name=YOUR_KEY_NAME"
```

notes:

- `ssh_key_name` is the key label from the lambda console (not a filepath).
- by default it picks the cheapest available 1-gpu instance.
- you can override the command like this:

```bash
terraform apply \
  -var "ssh_key_name=YOUR_KEY_NAME" \
  -var "run_cmd=torchrun --nproc_per_node=1 src/train.py --stage 3 --steps 200"
```

- you can request more gpus like this:

```bash
terraform apply \
  -var "ssh_key_name=YOUR_KEY_NAME" \
  -var "min_gpus=2"
```

## what we’ll plot for the paper

- peak memory vs throughput for each stage
- throughput vs bandwidth for each stage
- comm volume per step for each stage
