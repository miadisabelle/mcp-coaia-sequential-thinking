# GitHub Actions Integration

To integrate `co-lint` into your GitHub Actions workflow, create a new workflow file (e.g., `.github/workflows/colint.yml`) with the following content:

```yaml
name: CO-Lint
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install co-lint  # or python -m pip install ./.tools/co-lint
      - run: co-lint --config .co-lint.json --format github --fail-on error
```

This workflow will run on every push and pull request, automatically checking for compliance with the creative orientation guidelines and failing the check if errors are found.
