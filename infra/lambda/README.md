# lambda + terraform

this uses terraform as a thin wrapper around the lambda cloud cli (lai). it spins up a vm, runs one command, then tears it down.

## prereqs

- a lambda cloud account
- an ssh key added in the lambda console (note the key name)
- a cloud api token
- terraform installed locally

## quick start (fresh user)

1) add your ssh key in the lambda console and note the key name.
2) create an api token in the lambda console.
3) from this repo:

```bash
export LAMBDA_CLOUD_API_TOKEN="your_token_here"
cd infra/lambda
terraform init
terraform apply -var "ssh_key_name=YOUR_KEY_NAME"
```

## options

pick the cheapest available 1-gpu instance by default.

if you want a specific type:

```bash
terraform apply \
  -var "ssh_key_name=YOUR_KEY_NAME" \
  -var "instance_type=gpu_1x_a10"
```

change the run command like this:

```bash
terraform apply \
  -var "ssh_key_name=YOUR_KEY_NAME" \
  -var "run_cmd=torchrun --nproc_per_node=1 src/train.py --stage 3 --steps 200"
```

request more gpus:

```bash
terraform apply \
  -var "ssh_key_name=YOUR_KEY_NAME" \
  -var "min_gpus=2"
```

## notes

- this uses `lai run --rm`, so it stops the vm when the command finishes.
- you can re-run `terraform apply` to launch a fresh run.
