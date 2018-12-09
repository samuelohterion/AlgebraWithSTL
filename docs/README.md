# AlgebraWithSTL
Some algebra with std::vector<T> and std::vector<std::vector<T>>

## What's this about
This is a small collection of some functions and operator overloadings for convenient work  
with STL std::vector<T> and std::vector<std::vector<T>> as mathematical objects.  

## Whetting your appetite!   
There are some small neuronal network examples  

## How to build and run?
### Qt-Creator
  - Load the Qt-Project *AlgebraWithSTL.pro* in Qt-Creator!
  - Press Play! 

### Console
  - Open a console and go into the source direcory AlgebraWithSTL!
  - Run *make*!  
    ```..../[myCPPProjects]/AlgebraWithSTL$ make```

## Enjoy the Output! 
1. Start it!
2. Watch it!
3. Use it!

## How to use?
algebra.hpp and codeprinter.hpp only has to be included in your cpp file like this  
```
#include "algebra.hpp"
```
after that you can use all that you've seen in the demo AlgebraWithSTL like
```

Mat<double> zeta = {
  { 1. / 2., 0.,      0.      },
  { 0.,      2. / 3., 0.      },
  { 0.,      0.,      3. / 5. }
};

Vec<double>
  primes    = { 2, 3, 5 },
  positions = zeta | primes;

std::cout << "primes:\n" << primes << std::endl;
std::cout << "positions:\n" << positions << std::endl;
std::cout << "scalar product of primes and their positions is also prime! crazy!\n"
  << ( positions | primes ) 
  << std::endl;
```
or
```
Vec<int>
  v = {0,1,0,1,0,1,0,1};
  v = 2 * v - 1;

Mat<int> checkerboard = ( 1 + ( v ^ -v ) ) / 2;

std::cout << "checkerboard:\n" << checkerboard << std::endl;
```