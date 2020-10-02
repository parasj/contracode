all: black lint

black:
	black --line-length 140 representjs
	black --line-length 140 scripts

lint:
	flake8 --max-line-length 200 --ignore=E203 representjs