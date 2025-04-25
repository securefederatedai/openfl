metadata description = 'Create a resource group with the given name and region.'

@description('The name of the resource group.')
param resourceGroupName string

@description('The location.')
param location string

@description('Created on (timestamp)')
param createdOn string = utcNow('u')

targetScope = 'subscription'

resource resource_group 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: resourceGroupName
  location: location
  tags: {
    createdOn: createdOn
  }
}
