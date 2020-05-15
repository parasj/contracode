all: black lint

black:
	black --line-length 120 representjs

lint:
	flake8 representjs
