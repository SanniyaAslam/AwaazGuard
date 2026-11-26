variable "subscription_id" {
  description = "Azure for Students subscription ID (az account show --query id -o tsv)"
  type        = string
}

variable "project" {
  type    = string
  default = "awaazguard"
}

variable "location" {
  description = "Azure region. Student subscriptions only allow some regions; if apply says 'RequestDisallowedByAzure', try another (e.g. centralindia, uaenorth, southeastasia, eastus)."
  type        = string
  default     = "centralindia"
}

variable "vm_size" {
  description = "Standard_B2ats_v2 (2 vCPU, 1 GiB, x86) is in the free 750 h/month offer. Standard_B1s is the fallback."
  type        = string
  default     = "Standard_B2ats_v2"
}

variable "admin_username" {
  type    = string
  default = "azureuser"
}

variable "ssh_public_key_path" {
  description = "Public key created with: ssh-keygen -t ed25519 -f ~/.ssh/awaazguard"
  type        = string
  default     = "~/.ssh/awaazguard.pub"
}

variable "ssh_source" {
  description = "Who may reach SSH. '*' is needed for GitHub Actions deployments (login is key-only)."
  type        = string
  default     = "*"
}

variable "admin_cidr" {
  description = "Your public IP in CIDR form for Grafana, e.g. 39.45.10.20/32"
  type        = string
}

variable "dns_label" {
  description = "Unique lowercase name -> http://<dns_label>.<region>.cloudapp.azure.com"
  type        = string
}

variable "repo_url" {
  description = "HTTPS URL of your public GitHub repo"
  type        = string
}

variable "auto_shutdown_time" {
  description = "Optional daily shutdown, HHMM Pakistan time, e.g. \"0100\". Not needed: 750 free hours cover a VM running all month."
  type        = string
  default     = ""
}

variable "budget_email" {
  type    = string
  default = ""
}

variable "monthly_budget_usd" {
  type    = number
  default = 10
}
