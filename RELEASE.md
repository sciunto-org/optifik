
# Step by step release process


* Write changelog

* Push version

```
bumpver update --major	# MAJOR (breaking changes)	1.0.0 → 2.0.0
bumpver update --minor	# MINOR (nouvelles fonctionnalités)	1.0.0 → 1.1.0
bumpver update --patch	# PATCH (correctifs)	1.0.0 → 1.0.1
bumpver update --pre alpha	# Version alpha (1.0.0a1)	1.0.0 → 1.0.0a1
bumpver update --pre beta	# Version beta (1.0.0b1)	1.0.0 → 1.0.0b1
bumpver update --pre rc  # Version release candidate (1.0.0rc1)	1.0.0 → 1.0.0rc1
```


* Build
```
python -m build
```

* Check
```
twine check dist/*
```

* Push to pipy
```
twine upload dist/*
```
