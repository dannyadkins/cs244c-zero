# handy outputs for quick ssh
output "instance_name" {
  value = google_compute_instance.zero.name
}

output "instance_ip" {
  value = google_compute_instance.zero.network_interface[0].access_config[0].nat_ip
}
