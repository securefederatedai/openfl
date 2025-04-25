metadata description = 'Create compute resources for the deployment.'

@description('SSH public key for the deployed machines.')
param adminPublicKey string

@description('SSH username for the deployed machines.')
param adminUsername string = 'openfluser'

@description('Unique identifier for the deployment.')
param deploymentName string

@description('Total nodes to be deployed for the setup.')
param instanceCount string

@description('Size of the instances to be deployed.')
param instanceSize string

@description('Type of disk to be attached to the instances.')
param osDiskType string 

@description('Location where the setup is to be deployed.')
param location string

@description('Whether to deploy spot instances for the deployment.')
param useSpotVms bool = bool('false')

@description('Size of the OS Disk attached to the VMs.')
param osDiskSize int = 30

var customData = base64('${loadTextContent('user-data.txt')}')

// Resource ID of the delpoyed NSG.
var nsgId = resourceId('Microsoft.Network/networkSecurityGroups', 'nsg-${deploymentName}')

// reosurce ID of the default subnet for the deployed virtual network.
var subnetId = '${resourceId('Microsoft.Network/virtualNetworks', 'vnet-${deploymentName}')}/subnets/default'

// User-assigned Managed Identity to be set as identity for the VMSS, required for sending metrics
// to Azure-managed Prometheus.
resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'uami-${deploymentName}'
  location: location
}

// Virtual Machine Scale Set for the deployment. 
resource vmss 'Microsoft.Compute/virtualMachineScaleSets@2023-07-01' = {
  name: 'vmss-${deploymentName}'
  location: location
  sku: {
    name: instanceSize
    capacity: int(instanceCount)
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities:{
      '${uami.id}': {}
    }
  }
  properties: {
    virtualMachineProfile: {
      storageProfile: {
        osDisk: {
          createOption: 'fromImage'
          caching: 'ReadWrite'
          diskSizeGB: osDiskSize
          managedDisk: {
            storageAccountType: osDiskType
          }
        }
        imageReference: {
          publisher: 'canonical'
          offer: '0001-com-ubuntu-server-jammy'
          sku: '22_04-lts-gen2'
          version: 'latest'
        }
      }
      networkProfile: {
        networkInterfaceConfigurations: [
          {
            name: 'nic-${deploymentName}'
            properties: {
              primary: true
              enableAcceleratedNetworking: false
              disableTcpStateTracking: false
              networkSecurityGroup: {
                id: nsgId
              }
              dnsSettings: {
                dnsServers: [
                  
                ]
              }
              enableIPForwarding: false
              ipConfigurations: [
                {
                  name: 'ipconfig-${deploymentName}'
                  properties: {
                    publicIPAddressConfiguration: {
                      name: 'pip-${deploymentName}'
                      properties: {
                        idleTimeoutInMinutes: 15
                        ipTags: [
                          
                        ]
                        publicIPAddressVersion: 'IPv4'
                      }
                    }
                    primary: true
                    subnet: {
                      id: subnetId
                    }
                    privateIPAddressVersion: 'IPv4'
                  }
                }
              ]
            }
          }
        ]        
      }
      osProfile: {
        computerNamePrefix: 'vm-${deploymentName}'
        adminUsername: adminUsername
        customData: customData
        linuxConfiguration: {
          disablePasswordAuthentication: true
          ssh: {
            publicKeys: [
              {
                path: '/home/${adminUsername}/.ssh/authorized_keys'
                keyData: adminPublicKey
              }
            ]
          }
        }
      }
    }
    orchestrationMode: 'Uniform'
    platformFaultDomainCount: 1
    upgradePolicy: {
      mode: 'Manual'
    }
  }
}


