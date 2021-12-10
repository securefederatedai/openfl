# How to update the documentation

We use sphinx to generate the documentation for this project.
The documentation project has been initialized properly and we basically just need to update the actual content.

Install requirements for building documentation:

```sh
pip install -r requirements-docs.txt
```


The Makefile supports many targets. We choose html because we can easily host the documentation on a remote server. Compile the documentation source code:
```sh
make clean
make html
```

Open documentation locally:
```sh
cd _build/html
python -m http.server
```
