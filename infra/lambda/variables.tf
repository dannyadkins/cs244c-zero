variable "ssh_key_name" {
  type        = string
  description = "ssh key name in the lambda cloud console"
}

variable "min_gpus" {
  type        = number
  description = "min gpu count if instance_type is not set"
  default     = 1
}

variable "instance_type" {
  type        = string
  description = "lambda instance type (optional)"
  default     = ""
}

variable "run_cmd" {
  type        = string
  description = "command to run inside the vm"
  default     = "torchrun --nproc_per_node=1 src/train.py --stage 3"
}
