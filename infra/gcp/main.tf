# one-shot gpu vm for running a single experiment
terraform {
  required_version = ">= 1.3.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_instance" "zero" {
  # single vm runner
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone
  labels       = var.labels

  boot_disk {
    initialize_params {
      image = "projects/${var.image_project}/global/images/family/${var.image_family}"
      size  = var.disk_size_gb
      type  = var.disk_type
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    # preemptible if you want spot pricing
    on_host_maintenance = "TERMINATE"
    automatic_restart  = false
    preemptible        = var.use_spot
  }

  service_account {
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  metadata = {
    # these get read by startup.sh
    repo_url              = var.repo_url
    repo_ref              = var.repo_ref
    run_cmd               = var.run_cmd
    auto_shutdown_minutes = tostring(var.auto_shutdown_minutes)
    gcs_bucket            = var.gcs_bucket
  }

  metadata_startup_script = file("${path.module}/startup.sh")

  # keep it easy to destroy
  deletion_protection = false
}
