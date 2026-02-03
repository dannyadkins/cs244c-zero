project_id = "arboreal-inn-486300-j0"
region     = "us-central1"
zone       = "us-central1-a"

# l4 gpu setup (2x) — good price/perf for experiments
machine_type = "g2-standard-24"
gpu_type     = "nvidia-l4"
gpu_count    = 2

# gpu-ready ubuntu image (nvidia drivers preinstalled)
image_project = "ubuntu-os-accelerator-images"
image_family  = "ubuntu-accelerator-2204-amd64-with-nvidia-580"

# todo: set this to your public git repo url
repo_url = "https://github.com/dannyadkins/cs244c-zero.git"
repo_ref = "main"

use_spot = false

auto_shutdown_minutes = 240

run_cmd = "torchrun --nproc_per_node=2 src/train.py --stage 3"
