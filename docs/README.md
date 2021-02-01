# How to update the documentation

We use sphinx to generate the documentation for this project.
The documentation project has been initialized properly and we basically just need to update the actual content.

If we ever change the code structure since last compilation, we may need to regenerate the docstring index:
```shell
sphinx-apidoc -f -o . ../openfl
sphinx-apidoc -f -o . ../models
```

The command detects the code structure under `../openfl` and generates a series of `*.rst` files, such as `openfl.aggregator.rst`.
However, the docstring would not be compiled until we execute `make html` later.

We can also update the hand-crafted documents, including `intro.rst` and `tutorial.rst`. The `index.rst` is the entry file. We don't need to modify it unless we want to add more hand-crafted pages or adjust the order in the Contents page.


After completing revision on the .rst files, we would compile the documentation source code:
```
make clean
make html
```

The Makefile supports many targets. We choose html because we can easily host the documentation on a remote server:

```shell
cd _build/html
python -m http.server
```
