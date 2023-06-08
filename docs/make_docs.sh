#!/bin/bash
rm *.html
make html
rm -r stubs
cp _build/html/*.html ./
cp -r _build/html/stubs ./

