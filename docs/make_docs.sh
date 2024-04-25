rm *.html
make clean
make html
rm -r stubs
cp -r build/html/* ./
cp source/readme.rst ../
cp -r build/html/stubs ./
cp index.html ../
