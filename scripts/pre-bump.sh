#!/bin/bash
set -euo pipefail
uv lock
git add uv.lock
