.PHONY : run

run :	AWSTL
	./AWSTL

AWSTL: main.cpp algebra.hpp codeprinter.hpp
	g++ main.cpp -o AWSTL

