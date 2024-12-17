# Update documentation

To rebuild documentation, install packages:
```sh
pip install -r docs/requirements.txt
```

Compile the documentation source code:
```sh
make clean
make html
```

Serve documentation locally:
```sh
python -m http.server --directory _build/html
```
