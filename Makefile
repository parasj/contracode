all: black lint

black:
	black --line-length 120 representjs
	black --line-length 120 scripts

lint:
	flake8 --max-line-length 200 representjs

