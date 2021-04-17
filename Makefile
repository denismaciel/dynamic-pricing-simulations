
docs-update: nb-sync-notebooks nb-make-markdown
	mkdocs build

# nb-sync-notebooks:
# 	./scripts/compile_notebooks.py sync_notebooks

# nb-make-markdown:
# 	rm -rf docs/notebooks/*
# 	./venv/bin/python ./scripts/compile_notebooks.py make_markdown

install-dev-dependencies:
	pip install pip-tools
	pip-sync requirements.txt
	pip install -Ie .
	pre-commit install
