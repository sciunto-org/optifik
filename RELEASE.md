# Step by step release process

* Write changelog

* Commit changes

* Bump version

```
uv run bumpver update --major   # MAJOR (breaking changes)   1.0.0 -> 2.0.0
uv run bumpver update --minor   # MINOR (new features)       1.0.0 -> 1.1.0
uv run bumpver update --patch   # PATCH (fixes)              1.0.0 -> 1.0.1
uv run bumpver update --pre alpha    # alpha (1.0.0a1)       1.0.0 -> 1.0.0a1
uv run bumpver update --pre beta     # beta (1.0.0b1)        1.0.0 -> 1.0.0b1
uv run bumpver update --pre rc       # release candidate     1.0.0 -> 1.0.0rc1
```

* Build and publish

```
uv run twine check dist/* && uv run twine upload dist/*
```


## Details


* Build

```
uv build
```

* Check

```
uv run twine check dist/*
```

* Push to PyPI

```
uv run twine upload dist/*
```
