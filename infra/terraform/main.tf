# AwaazGuard on Azure for Students - every resource here is covered by the free offer:
#   * 1 Linux VM Standard_B2ats_v2 (750 free hours/month for 12 months; renews while you study)
#   * 1 Premium SSD P6 (64 GB) OS disk (free amount for 12 months)
#   * 1 static public IP, network, firewall rules (cents/month, paid from the $100 credit)
#   * a budget alert so you are emailed long before the credit could run out
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 4.0" }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.project}-rg"
  location = var.location
  tags     = { project = var.project }
}

# ------------------------------------------------------------------ network
resource "azurerm_virtual_network" "vnet" {
  name                = "${var.project}-vnet"
  address_space       = ["10.20.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_subnet" "subnet" {
  name                 = "default"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.20.1.0/24"]
}

resource "azurerm_public_ip" "ip" {
  name                = "${var.project}-ip"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
  domain_name_label   = var.dns_label # gives <label>.<region>.cloudapp.azure.com
}

resource "azurerm_network_security_group" "nsg" {
  name                = "${var.project}-nsg"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  security_rule {
    name                       = "web"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_ranges    = ["80", "443"]
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  security_rule {
    name                       = "ssh"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.ssh_source # "*" lets GitHub Actions deploy over SSH (key-only login)
    destination_address_prefix = "*"
  }
  security_rule {
    name                       = "grafana"
    priority                   = 120
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "3001"
    source_address_prefix      = var.admin_cidr
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_interface" "nic" {
  name                = "${var.project}-nic"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.ip.id
  }
}

resource "azurerm_network_interface_security_group_association" "nic_nsg" {
  network_interface_id      = azurerm_network_interface.nic.id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

# ------------------------------------------------------------------ virtual machine
resource "azurerm_linux_virtual_machine" "vm" {
  name                  = "${var.project}-vm"
  location              = azurerm_resource_group.rg.location
  resource_group_name   = azurerm_resource_group.rg.name
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.nic.id]

  admin_ssh_key {
    username   = var.admin_username
    public_key = file(pathexpand(var.ssh_public_key_path))
  }
  disable_password_authentication = true

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 64 # P6: the size included in the free offer
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server"
    version   = "latest"
  }

  custom_data = base64encode(templatefile("${path.module}/cloud-init.yaml", {
    repo_url = var.repo_url
    user     = var.admin_username
  }))

  tags = { project = var.project }
}

# Shut the VM down every night so it never burns hours you are not using (optional).
resource "azurerm_dev_test_global_vm_shutdown_schedule" "nightly" {
  count                 = var.auto_shutdown_time == "" ? 0 : 1
  virtual_machine_id    = azurerm_linux_virtual_machine.vm.id
  location              = azurerm_resource_group.rg.location
  enabled               = true
  daily_recurrence_time = var.auto_shutdown_time
  timezone              = "Pakistan Standard Time"
  notification_settings { enabled = false }
}

# ------------------------------------------------------------------ budget alert
resource "azurerm_consumption_budget_resource_group" "budget" {
  count             = var.budget_email == "" ? 0 : 1
  name              = "${var.project}-budget"
  resource_group_id = azurerm_resource_group.rg.id
  amount            = var.monthly_budget_usd
  time_grain        = "Monthly"

  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00Z", timestamp())
  }

  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    contact_emails = [var.budget_email]
  }

  lifecycle { ignore_changes = [time_period] }
}
