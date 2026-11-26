output "public_ip" { value = azurerm_public_ip.ip.ip_address }
output "app_url" { value = "http://${azurerm_public_ip.ip.fqdn}" }
output "grafana_url" { value = "http://${azurerm_public_ip.ip.fqdn}:3001" }
output "fqdn" { value = azurerm_public_ip.ip.fqdn }
output "ssh_command" { value = "ssh -i ~/.ssh/awaazguard ${var.admin_username}@${azurerm_public_ip.ip.ip_address}" }
output "vm_name" { value = azurerm_linux_virtual_machine.vm.name }
output "resource_group" { value = azurerm_resource_group.rg.name }
