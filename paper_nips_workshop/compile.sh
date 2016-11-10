#!/bin/bash
pdflatex nips_2016.tex
bibtex nips_2016
pdflatex nips_2016.tex
bibtex nips_2016
pdflatex nips_2016.tex
bibtex nips_2016
evince nips_2016.pdf
