rm *.html
rm -r _sources
rm -r _static
rm -r _modules
rm -r doctrees
rm -r build
rm -r source/stubs
rm source/modules.rst
rm source/celltraj.rst
sphinx-apidoc -o ./source ../celltraj
make clean
make html
#sphinx-build -M html source/ build/ -a -j auto -n --keep-going
cp -r build/html/* ./
cp -r build/ ./
cp source/readme.rst ../
cp index.html ../
