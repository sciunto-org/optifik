
# Step by step release process

* Ensure packages are installed
```
pip install build twine
```

* Write changelog

* Commit changes

* Push version

```
bumpver update --major	# MAJOR (breaking changes)	1.0.0 → 2.0.0
bumpver update --minor	# MINOR (new features)	1.0.0 → 1.1.0
bumpver update --patch	# PATCH (fixes)	1.0.0 → 1.0.1
bumpver update --pre alpha	# alpha (1.0.0a1)	1.0.0 → 1.0.0a1
bumpver update --pre beta	# beta (1.0.0b1)	1.0.0 → 1.0.0b1
bumpver update --pre rc  # release candidate (1.0.0rc1)	1.0.0 → 1.0.0rc1
```

```
git push origin --tag && python -m build && twine check dist/* && twine upload dist/*
```


## Details


* Push tags

```
git push origin --tag
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
