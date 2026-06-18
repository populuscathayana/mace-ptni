# PtNi MACE Workflow

This is the refactored workflow tree for the PtNi MACE project.  Historical
scripts and documents remain under `outputs/`; new runs should start from this
directory and write to `mace_workspace/`.

Main documentation:

```text
ptni_mace_workflow/docs/README_中文.md
```

Build the static documentation site:

```bash
python ptni_mace_workflow/tools/build_docs_site.py
```

Open:

```text
mace_workspace/reports/docs_site/index.html
```
