## Cfssl PKI workflow.

### 1. Create CA (on CA side).
```
  step ca init --password-file pass_file

    os.system(f'./{step} ca provisioner remove prov --all')
    os.system(f'./{step} crypto jwk create {step_config_dir}/certs/pub.json '
              + f'{step_config_dir}/secrets/priv.json --password-file={pki_dir}/pass_file')
    os.system(f'./{step} ca provisioner add provisioner {step_config_dir}/certs/pub.json')
    echo('Up CA server')
  cfssl gencert -initca csr_ca.json
```
get  ```ca-key.pem  ca.pem```
### 2. Generate server key pair (on CA side) to establish tls connection between CA and clients (sign it by our CA).
```
  cfssl gencert -ca=ca.pem -ca-key=ca-key.pem  -hostname='localhost' csr_server.json
```
get  ```server-key.pem  server.pem```
### 3. Generate auth key (on CA side).
```
echo -n $(openssl rand -hex 16 | tr -d '\n') > base.key
```
get  ```base.key: CFD0F71279D67A0E4B826A2528FE7487```

### 4. Up HTTPS CA server (on CA side).
```
  cfssl serve -ca-key ca-key.pem -ca ca.pem -tls-key server-key.pem -tls-cert server.pem -config config_ca.json
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
### 5, 6. Request CA certificate from CA server (on aggregator and collaborator side).
In untrusted area: manually deliver CA cert from CA to aggregator or collaborator.<br>
In trusted area use:
```
cfssl info -remote ca_host:8888
```
get ```ca.pem```
### 7.a Generate key pair for aggregator
### 7.b Sign generated aggregator cert. 
One command for a and b steps:
```
cfssl gencert -hostname='host' -tls-remote-ca ca.pem -config config_server.json csr_server.json
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
    "caserver" : "htpps://ca_host:8888"
  }
}
```
### 8a, 8b. Similarly for the collaborator (on collaborator side).
```
cfssl gencert -tls-remote-ca ca.pem -config config_client.json csr_client.json
```
get ```col-key.pem  col.pem```

### Note: Trusted area.
If you run this in trusted area, you don`t need to use https server, so run all this commands without ```-tls-key -tls-cert -tls-remote-ca``` keys and skip step 2<br><br>


![Cfssl workflow](./images/cfssl_flow.svg?raw=true "Cfssl workflow")
