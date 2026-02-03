# basic project + vm config
variable "project_id" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type        = string
  description = "GCP region"
  default     = "us-central1"
}

variable "zone" {
  type        = string
  description = "GCP zone"
  default     = "us-central1-a"
}

variable "instance_name" {
  type        = string
  description = "VM name"
  default     = "zero-runner"
}

variable "machine_type" {
  type        = string
  description = "GCP machine type. Pick one that supports your GPU choice."
}

variable "gpu_type" {
  type        = string
  description = "GCP GPU accelerator type (e.g., A100, L4, T4)."
}

variable "gpu_count" {
  type        = number
  description = "Number of GPUs"
  default     = 2
}

variable "disk_size_gb" {
  type        = number
  description = "Boot disk size"
  default     = 200
}

variable "disk_type" {
  type        = string
  description = "Boot disk type"
  default     = "pd-ssd"
}

variable "image_project" {
  type        = string
  description = "Image project for your GPU image"
}

variable "image_family" {
  type        = string
  description = "Image family for your GPU image"
}

variable "use_spot" {
  type        = bool
  description = "Use preemptible/spot capacity"
  default     = false
}

variable "auto_shutdown_minutes" {
  type        = number
  description = "Safety timer in minutes before the VM powers off"
  default     = 240
}

variable "repo_url" {
  type        = string
  description = "Git repo URL to clone"
}

variable "repo_ref" {
  type        = string
  description = "Branch or commit to checkout"
  default     = "main"
}

variable "run_cmd" {
  type        = string
  description = "Command to run on the VM"
  default     = ""
}

variable "gcs_bucket" {
  type        = string
  description = "Optional GCS bucket (gs://...) to upload logs"
  default     = ""
}

variable "labels" {
  type        = map(string)
  description = "Optional instance labels"
  default     = {}
}
