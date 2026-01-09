pre-commit-install:
	pre-commit install --config config/pre-commit-config.yaml

pre-commit-run:
	pre-commit run --all-files --config config/pre-commit-config.yaml
pre-commit-setup:
	pre-commit-install pre-commit-run
