# GCP one-shot runner

This spins up a single GPU VM, runs a command, then shuts down. It is meant to be easy to destroy so you don't accidentally keep paying.

## Why this is safe-ish

- The VM schedules an automatic shutdown after `auto_shutdown_minutes`.
- It shuts down again after the run finishes.
- You can still `terraform destroy` to clean up disks and IPs.

## Prereqs

- `gcloud` installed and logged in
- `terraform` installed
- A GCP project with GPU quota

## Quick usage

```bash
cd infra/gcp
terraform init
```

Then apply with your project details:

```bash
terraform apply \
  -var "project_id=YOUR_PROJECT" \
  -var "zone=YOUR_ZONE" \
  -var "machine_type=YOUR_MACHINE_TYPE" \
  -var "gpu_type=YOUR_GPU_TYPE" \
  -var "gpu_count=2" \
  -var "image_project=YOUR_IMAGE_PROJECT" \
  -var "image_family=YOUR_IMAGE_FAMILY" \
  -var "repo_url=YOUR_REPO_URL" \
  -var "run_cmd=torchrun --nproc_per_node=2 src/train.py --stage 3" \
  -var "auto_shutdown_minutes=240"
```

To tear it down:

```bash
terraform destroy
```

## Finding GPU-compatible values

- Use `gcloud compute accelerator-types list --zones YOUR_ZONE` for GPU types.
- Use `gcloud compute machine-types list --zones YOUR_ZONE` for machine types.
- For images, pick a GPU-ready image family and project.

## Notes

- If you don't destroy the VM, shutdown keeps costs low but disks can still bill a bit.
- Set `use_spot=true` to use preemptible capacity for cheaper runs.
