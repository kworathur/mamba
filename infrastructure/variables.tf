variable "project_id" {
    description = "ID of the google cloud project"
    type = string
}

variable "region" {
    description = "GCP region"
    type = string
}

variable "zone" {
    description = "GCP zone"
    type = string
}

variable "client_ip" {
    description = "Client IP address for ingress SSH firewall rule"
    type = string
}