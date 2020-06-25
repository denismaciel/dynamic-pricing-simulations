
docs-update: nb-sync-notebooks nb-make-markdown
	mkdocs build

nb-execute-notebooks:
	echo "Not implemented" && exit 1

nb-sync-notebooks:
	./scripts/compile_notebooks.py sync_notebooks

nb-make-markdown:
	./scripts/compile_notebooks.py make_markdown
