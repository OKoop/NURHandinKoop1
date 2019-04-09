#!/bin/bash

echo "Run handin s1676059 ..."

#Download the five satgals files for use in Exercise 3
echo "Download files for use in report..."
if [ ! -e sine.png ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m15.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m14.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m13.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m12.txt
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m11.txt
fi

#Runs the python script and thus creates all plots and saves outputs to a txt-file
echo "Run the python script ..."
python3 HIv1.py > outputs.txt

#Generate the pdf by using LaTeX
echo "Generating the pdf ..."
pdflatex HI1.tex
bibtex HI1.aux
pdflatex HI1.tex
pdflatex HI1.tex


