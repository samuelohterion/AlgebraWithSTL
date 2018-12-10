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
Just include algebra.hpp and codeprinter.hpp in your cpp file.  
No linking against some library is neccessary.
```
#include "algebra.hpp"
#include "codeprinter.hpp"
```
#### codeprinter usage
```
// One creates a CodePrinter object with path to main.cpp file that should be shown

CodePrinter cp( "../AlgebraWithSTL/main.cpp" );

// print block "examples"
cp.print( "examples" );

// a block
// starts with //@BlockName
// and ends with //@

// so the "examples" block
// starts with //@examples
// and ends with //@

//@examples
  // Examples for using algebra.hpp
  // -------- --- ----- -----------
  // next with [ENTER]
  // exit with x or X or q or Q and [ENTER] + [ENTER]
  // do something nice here!
//@
  // WFE wait for enter, can be found in the main.cpp, also the print() function
CodePrinter::WFE( );

// now the same with block "create a vector"
cp.print( "create a vector" );
//@create a vector
  VD u = { 1., 2., 3., 4. };
  print( "u", u );
//@
CodePrinter::WFE( );

// same procedure with "operator vector"
cp.print( "operator vector" );
//@operator vector
  // some unary operators
  print( "+u", +u );
  print( "-u", -u );
//@
CodePrinter::WFE( );

...
```
#### algebra usage

This is too much to show here again.  
Watch the demo and everything will be clear!  
But 2 short examples (not in the demo):  
^ works as outer product  
| works as inner product like in  
Vector x Vector,  
Matrix x Vector,  
Vector x Matrix,  
Matrix x Matrix.  
Here Vector x Vector is the scalar product
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