metadata description = 'Create network resources for the deployment.'

@description('Unique identifier for the deployment.')
param deploymentName string

@description('Location where the setup is to be deployed.')
param location string

// Create a virtual network to host the VMSS nodes.
resource vnet 'Microsoft.Network/virtualNetworks@2023-09-01' = {
  name: 'vnet-${deploymentName}'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.1.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'default'
        properties: {
          addressPrefix: '10.1.0.0/20'
        }
      }
    ]
  }
}

// Intel NAT allowed IP list.
var allowedSourceIPs = [ '134.134.139.64/27', '134.134.137.64/27', '192.55.54.32/27', '192.55.55.32/27', '192.198.151.32/27', '134.191.227.32/27', '134.191.220.64/27', '134.191.221.64/27', '134.191.196.160/27', '134.191.197.160/27', '134.191.232.64/27', '134.191.233.192/27', '192.55.79.160/27', '198.175.68.32/27', '192.198.146.160/27', '192.198.147.160/27', '192.102.204.32/27', '192.55.46.32/27' ]

// Network security group with Intel NAT allowed IPs to protect SSH access.
resource nsg 'Microsoft.Network/networkSecurityGroups@2023-09-01' =  {
  name: 'nsg-${deploymentName}'
  location: location
  properties: {
    securityRules: [
      {
        name: 'default-allow-ssh'
        properties: {
          priority: 310
          protocol: 'TCP'
          access: 'Allow'
          direction: 'Inbound'
          sourceApplicationSecurityGroups: []
          destinationApplicationSecurityGroups: []
          sourceAddressPrefixes: allowedSourceIPs
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
    ]
  }
}

