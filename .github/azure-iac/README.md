# Azure infrastructure provisioning for OpenFL

## Steps

1. Login to Azure using az-cli:
```
az login
```
2. Select the right subscription:
```
az account set --subscription subscriptionId
```
3. Create the deployment:

## Parameters to create deployment
- `deploymentName`: Unique identifier for the deployment.
- `location`: Location where the setup is to be deployed.
  Defaults to `eastus`.
- `totalNodes`: Total nodes to be deployed for the setup.
  Defaults to 1.
- `instanceSize`: Size of the instances to be deployed.
  Defaults to `Standard_B2s`.
- `osDiskType`: Type of disk to be attached to the instances.
  Defaults to `StandardSSD_LRS`.
- `adminUsername`: SSH username for the deployed machines.
  Defaults to `openfluser`.
- `adminPublicKey`: SSH public key for the deployed machines.
- `osDiskSize`: Size of the OS disk attached to the VMs.
  Defaults to `30` GB.

Sample command
```
az deployment sub create -l eastus -f main.bicep \
  -p deploymentName=openfl-infra \
  -p location=eastus \
  -p totalNodes=1 \
  -p instanceSize=Standard_B2s \
  -p adminPublicKey='ssh-rsa AAA... generated-by-azure' \
  -p osDiskSize=50
```

4. To get the private IPs of the deployed VMs, use command.
```
az vmss nic list \
  --resource-group rg-{deploymentName} \
  --vmss-name vmss-{deploymentName} \
  --query "[].ipConfigurations[].privateIPAddress"
```

5. To get the public IPs of the deployed VMs, use command.
```
az vmss list-instance-public-ips \
  --resource-group rg-{deploymentName} \
  --name vmss-{deploymentName} \
  --query "[].ipAddress"
```
