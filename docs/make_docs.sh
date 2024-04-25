#!/bin/bash
rm *.html
make html
rm -r stubs
cp _build/html/*.html ./
cp -r _build/html/stubs ./
#pandoc -f html -t markdown readme.html -o README.md
#cp README.md ../
cp index.html ../

