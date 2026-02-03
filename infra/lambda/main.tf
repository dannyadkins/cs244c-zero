terraform {
  required_version = ">= 1.3.0"
}

resource "null_resource" "lambda_run" {
  triggers = {
    run_cmd       = var.run_cmd
    min_gpus      = tostring(var.min_gpus)
    instance_type = var.instance_type
    ssh_key_name  = var.ssh_key_name
  }

  provisioner "local-exec" {
    command = "${path.module}/../../scripts/lambda_run.sh"
    environment = {
      SSH_KEY_NAME  = var.ssh_key_name
      MIN_GPUS      = tostring(var.min_gpus)
      INSTANCE_TYPE = var.instance_type
      RUN_CMD       = var.run_cmd
    }
  }
}
