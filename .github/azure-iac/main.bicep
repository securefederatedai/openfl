metadata description = 'Create a setup to run an OpenFL federation'

targetScope = 'subscription'

@description('Unique identifier for the deployment.')
param deploymentName string

@description('Location where the setup is to be deployed.')
param location string = 'eastus'

@description('Total nodes to be deployed for the setup.')
param totalNodes string = '1'

@description('Size of the instances to be deployed.')
param instanceSize string = 'Standard_B2s'

@description('Type of disk to be attached to the instances.')
param osDiskType string = 'StandardSSD_LRS'

@description('SSH username for the deployed machines.')
param adminUsername string = 'openfluser'

@description('SSH public key for the deployed machines.')
param adminPublicKey string

@description('Size of the OS Disk attached to the VMs.')
param osDiskSize int = 30

// Resource group for the deployment.
var deploymentGroupName = 'rg-${deploymentName}'

// Deploy resource group for the deployment.
module resource_group 'resource_group.bicep' = {
  name: 'deploy-${deploymentGroupName}'
  scope: subscription()
  params: {
    location: location
    resourceGroupName: deploymentGroupName
  }
}

// Deploy network resources for the deployment.
module networking 'network.bicep' = {
  name: 'network-${deploymentName}'
  scope: resourceGroup(deploymentGroupName)
  dependsOn: [resource_group]
  params: {
    location: location
    deploymentName: deploymentName
  }
}

// Deploy compute resources for the deployment.
module  compute 'compute.bicep' = {
  name: 'compute-${deploymentName}'
  scope: resourceGroup(deploymentGroupName)
  dependsOn: [networking]
  params: {
    adminPublicKey: adminPublicKey
    adminUsername: adminUsername
    deploymentName: deploymentName
    instanceCount: totalNodes
    instanceSize: instanceSize
    location: location
    osDiskType: osDiskType
    osDiskSize: osDiskSize
  }
}
