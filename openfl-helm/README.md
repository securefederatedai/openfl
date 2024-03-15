# OpenFL helm chart

To deploy the OpenFL chart use:

```bash
helm install demo openfl-helm/ | tee helm.log
```

There is an example workflow described in the `NOTES` section after chart deployment.
The example is a `keras_cnn_mnist` template-based workflow with one aggregator and 2 collaborators.

## Running example flow in a cluster behind a proxy

If a proxy for an internet connection is required, set variables with desired proxy values
in steps:

- 1
- 3
- 10

inside the container. For example:

```bash
export http_proxy=http://proxy.example.com:1234
export https_proxy=http://proxy.example.com:1234
export no_proxy=.cluster.local
```

> no_proxy setting is required for the internal connection within a cluster

## Example deployment log

```bash
$ helm install example openfl-helm/ | tee helm.log
NAME: example
LAST DEPLOYED: Mon May 15 04:55:09 2023
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
Running example federated learning based on keras_cnn_mnist template:

1. Prepare the workspace.
$ kubectl exec -it example-openfl-aggregator-0 -- bash -c "
fx workspace create --prefix /openfl/my_federation --template keras_cnn_mnist
sed -i 's/    agg_port:.*/    agg_port: 9999/g' /openfl/my_federation/plan/plan.yaml
cd /openfl/my_federation ; fx plan initialize ; fx workspace certify ; fx aggregator generate-cert-request ; fx aggregator certify -s ; fx workspace export
"

2. Propagate workspace across federation members.
$ kubectl cp default/example-openfl-aggregator-0:/openfl/my_federation/my_federation.zip ./my_federation.zip
for x in {0..1}; do
kubectl cp ./my_federation.zip default/example-openfl-collaborator-${x}:/openfl/my_federation.zip
done

3. Initialize collaborators.
$ for x in {0..1}; do
kubectl exec -it example-openfl-collaborator-${x} -- bash -c "
export COLLABORATOR_ID=\$(hostname | awk -F- '{print \$NF}')
cd /openfl ; fx workspace import --archive my_federation.zip ; cd my_federation ; fx collaborator generate-cert-request -s -n \${COLLABORATOR_ID}"
done

4. Copy signing requests from collaborators to the aggregator.
$ for x in {0..1}; do
REQUEST=col_${x}_to_agg_cert_request.zip
kubectl cp default/example-openfl-collaborator-${x}:/openfl/my_federation/$REQUEST $REQUEST
kubectl cp $REQUEST default/example-openfl-aggregator-0:/openfl/my_federation/$REQUEST
done

5. Sign requests on the aggregator.
$ kubectl exec -it example-openfl-aggregator-0 -- bash -c "
cd /openfl/my_federation
for x in {0..1}; do
fx collaborator certify -s --request-pkg col_\${x}_to_agg_cert_request.zip
done"

6. Copy signed requests from the aggregator to collaborators.
$ for x in {0..1}; do
REQUEST=agg_to_col_${x}_signed_cert.zip
kubectl cp default/example-openfl-aggregator-0:/openfl/my_federation/$REQUEST $REQUEST
kubectl cp $REQUEST default/example-openfl-collaborator-${x}:/openfl/my_federation/$REQUEST
done

7. Import signed certs on collaborators.
$ for x in {0..1}; do
kubectl exec -it example-openfl-collaborator-${x} -- bash -c "
export COLLABORATOR_ID=\$(hostname | awk -F- '{print \$NF}')
cd /openfl/my_federation ; fx collaborator certify --import agg_to_col_\${COLLABORATOR_ID}_signed_cert.zip"
done

8. Start a tmux / screen session with 3 panes. Run each of the components below in a separate pane.

9. Start the aggregator.
$ kubectl exec -it example-openfl-aggregator-0 -- bash -c "
cd /openfl/my_federation ; fx aggregator start"

10. Start the collaborators. Repeat the steps from below for N={0..1} manually, each in a separate pane.
$ kubectl exec -it example-openfl-collaborator-${N} -- bash -c "
export COLLABORATOR_ID=\$(hostname | awk -F- '{print \$NF}')
sed -i \"s|\${COLLABORATOR_ID},data/\${COLLABORATOR_ID}|\${COLLABORATOR_ID},\${COLLABORATOR_ID}|\" /openfl/my_federation/plan/data.yaml
cd /openfl/my_federation ; fx collaborator start -n \${COLLABORATOR_ID}"
```
