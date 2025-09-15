# Pre-commit Integration

To integrate `co-lint` with your pre-commit hooks, add the following to your `.pre-commit-config.yaml` file:

```yaml
repos:
  - repo: local
    hooks:
      - id: co-lint
        name: creative-orientation-lint
        entry: co-lint
        language: system
        types: [markdown]
        files: ^(specifications/|agents/|[^/]*README.md|CLAUDE.md|GEMINI.md|CURSOR.md)
        args: ["--config", ".co-lint.json", "--format", "github", "--fail-on", "error"]
```

This will run `co-lint` on every commit, checking only the specified markdown files and providing feedback directly in your terminal.
