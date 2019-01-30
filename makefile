.PHONY : run

run :	AlgebraWithSTL
	./AlgebraWithSTL
	rm AlgebraWithSTL
	rm weights

AlgebraWithSTL: main.cpp algebra.hpp codeprinter.hpp
	g++ main.cpp -o AlgebraWithSTL

