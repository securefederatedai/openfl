## Cfssl PKI workflow.

### 1. Create CA (on CA side).
```
  cfssl gencert -initca csr_ca.json

```
get  ```ca-key.pem  ca.pem```
### 2. Generate auth key (on CA side).
```
echo -n $(openssl rand -hex 16 | tr -d '\n') > base.key
```
get  ```base.key: CFD0F71279D67A0E4B826A2528FE7487```

### 3. Up CA server (on CA side).
```
  cfssl serve -ca-key ca-key.pem -ca ca.pem -config config_ca.json
```
Such command uses config_ca.json file:
```json
{
  "signing": {
    "default": {
      "auth_key": "key1",
      "expiry": "8760h",
      "usages": [
         "signing",
         "key encipherment",
         "server auth",
         "client auth"
       ]
     }
  },
  "auth_keys": {
    "key1": {
      "key": "file:base.key",
      "type": "standard"
    }
  }
}
```
```file:base.key``` - auth key which we have created
### 4. Generate and sign certificate and private key on agregator (on aggregator side). 
```
cfssl gencert -hostname='host' -config config_server.json csr_server.json
```
get ```agg-key.pem  agg.pem```

```config_server.json:```
```json
{
  "auth_keys" : {
    "key1" : {
        "type" : "standard",
        "key" : "CFD0F71279D67A0E4B826A2528FE7487"
    }
  },
  "signing" : {
    "default" : {
        "auth_remote" : {
          "remote" : "caserver",
          "auth_key" : "key1"
        }
    }
  },
  "remotes" : {
    "caserver" : "ca_host:8888"
  }
}
```
### 5. Similarly for the collaborator  (on collaborator side).
```
cfssl gencert -config config_client.json csr_client.json
```
get ```col-key.pem  col.pem```
### 6. Request CA certificate from CA server (on aggregator and collaborator side).
```
cfssl info -remote ca_host:8888
```
get ```ca.pem```
```mermaid
sequenceDiagram
Title: Collaborator Certificate Signing Flow
  participant A as Aggregator
  participant CA as CA
  participant C as Collaborator
  CA->>CA: 1. Create CA:<br>`cfssl gencert -initca csr_ca.json`
  CA->>CA: 2. Create auth key:<br>`echo -n $(openssl rand -hex 16 | tr -d '\n') > base.key` to generate auth_key
  CA->>CA: 3. Up CA server:<br>`cfssl serve -ca-key ca-key.pem -ca ca.pem -config config_ca.json` <br> to up CA server
  A->>CA: 4. Generate and sign aggregator cert:<br>`cfssl gencert -hostname='host' -config config_server.json csr_server.json`
  Note over A,CA: Get signed agg.pem 
  C->>CA: 5. Generate and sign collaborator cert:<br>`cfssl gencert -config config_client.json csr_client.json`
  Note over C,CA: Get signed col.pem 
  C->>CA: 6. Request CA cert:<br>`cfssl info -remote ca_host:8888`
  Note over C,CA: Get ca.pem
  A->>CA: 7. Request CA cert:<br>`cfssl info -remote ca_host:8888`
  Note over A,CA: Get ca.pem
  ```