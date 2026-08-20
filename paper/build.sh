#!/bin/sh
set -eu

paper_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$paper_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
