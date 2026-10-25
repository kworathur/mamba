# Provision one v5e chip (spot VM) for experiments and debugging
provider "google-beta" {
  project     = "${var.project_id}"
  region      = "${var.region}"
  zone = "${var.zone}"
}

data "google_tpu_v2_runtime_versions" "available" {
  provider = google-beta
}

data "google_tpu_v2_accelerator_types" "available" {
  provider = google-beta
}

resource "google_tpu_v2_vm" "tpu" {
  provider = google-beta

  name = "keshav-tpu"
  description = "Test TPU for vLLM experiments."

  runtime_version  = "v2-alpha-tpuv5-lite"

  accelerator_config {
    type     = "V5LITE_POD"
    topology = "1x1"
  }
  metadata = {
	ssh-keys = "keshavworathur_gmail_com:${file("~/.ssh/tpu_key.pub")}"
  }

  scheduling_config {
    preemptible = true
    spot = true
  }

  network_config {
    can_ip_forward      = true
    enable_external_ips = true
    network             = google_compute_network.network.id
    subnetwork          = google_compute_subnetwork.subnet.id
  }

  shielded_instance_config {
    enable_secure_boot = true
  }

  service_account {
    email = google_service_account.sa.email
    scope = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  data_disks {
    source_disk = google_compute_disk.disk.id                                                      
    mode        = "READ_ONLY"
  }

  depends_on = [time_sleep.wait_60_seconds]
}

resource "google_compute_subnetwork" "subnet" {
  provider = google-beta

  name          = "tpu-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = "${var.region}"
  network       = google_compute_network.network.id
}

resource "google_compute_network" "network" {
  provider = google-beta

  name                    = "tpu-net"                                                            
  auto_create_subnetworks = false
}

resource "google_compute_firewall" "allow_ssh" {
  provider  = google-beta                                                                        
  name      = "tpu-allow-ssh"
  network   = google_compute_network.network.id                                                  
  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = [var.client_ip] 
}

resource "google_service_account" "sa" {
  provider = google-beta

  account_id   = "tpu-sa"
  display_name = "TPU manager service account"
}

resource "google_project_iam_member" "tpu_admin_binding" {
  project = "${var.project_id}"
  role = "roles/tpu.admin"
  member = "serviceAccount:${google_service_account.sa.email}"
}

# prevent unexpected costs from VMs left running by accident
resource "google_cloud_scheduler_job" "tpu_deletion_job" {
  name = "delete-${google_tpu_v2_vm.tpu.name}-job"
  project = "${var.project_id}"
  region = "${var.region}"
  description = "Deletes spot TPU instance ${google_tpu_v2_vm.tpu.name} after time limit"
  schedule    = "* */4 * * *" # Runs every 4 hours
  time_zone   = "UTC"

  # Target the Cloud TPU v2 REST API to delete the specific TPU instance
  http_target {
    http_method = "DELETE"
    uri         = "https://tpu.googleapis.com/v2/projects/${var.project_id}/locations/${var.zone}/nodes/${google_tpu_v2_vm.tpu.name}"

    oauth_token {
      service_account_email = google_service_account.sa.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [
    google_tpu_v2_vm.tpu,
    google_project_iam_member.tpu_admin_binding
  ]
}


resource "google_compute_disk" "disk" {                                                          
  provider = google-beta

  name  = "tpu-disk"                                                                             
  image = "debian-cloud/debian-12"
  size  = 10                                                                                     
  type  = "pd-ssd"
  zone  = "${var.zone}"
}
# Wait after service account creation to limit eventual consistency errors.                    
resource "time_sleep" "wait_60_seconds" {
  depends_on = [google_service_account.sa]
  create_duration = "60s"
}
