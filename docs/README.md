# Compiling the Documentation

You can find this documentation <a href="https://safire.flatironinstitute.org/autohf/dev">online</a>. To compile it yourself, install autohf with the `DOCS` feature

```bash
$ pip install .[DOCS]
```
and compile the docs using `make` with,

```bash
$ cd /path/to/autohf/docs
$ make html
```

which will generate the documentation as html in the directory, `/path/to/autohf/docs/_build/html`.

