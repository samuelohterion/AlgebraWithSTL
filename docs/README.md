# AlgebraWithSTL
Some algebra with std::vector<T> and std::vector<std::vector<T>>

## What's this about
This is a small collection of some functions and operator overloadings for convenient work
with STL std::vector<T> and std::vector<std::vector<T>> as mathematical objects.

## Wetting your appetite!
There are some small neuronal network examples

## How to build and run?
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

Note:
^ works as outer product
| works as inner product like in
Vector times Vector,
Matrix times Vector,
Vector times Matrix,
Matrix times Matrix.
Vector times Vector is the scalar product

This is too much to show here again.
Watch the demo and everything will be clear!
But 2 short examples (not in the demo):

```
#include "algebra.cpp"

using namespace alg;
//@some typedefs for std::vector< T >
// just for abbr.
// vector
// template < typename T > using
// Vec = std::vector< T >;

//// matrix
// template < typename T > using
// Mat = std::vector< std::vector< T > >;

//// tensor 3rd degree
// template < typename T > using
// Tsr = std::vector< std::vector< std::vector< T > > >;

//// tensor 4rd degree
// template < typename T > using
// Tsr4 = std::vector< std::vector< std::vector< std::vector< T > > > >;

/*
typedef std::size_t   UI;
typedef Vec< UI >     VU;
typedef double         D;
typedef Vec< D >      VD;
typedef Mat< D >      MD;
typedef Tsr< D >      TD;
*/
//@
typedef std::complex< long double > CMPLX;

int
main() {

// Pauli Matrices
	// complex vector operator
	Tsr< CMPLX >
	sigma = {
		{
			{CMPLX(0.l, 0.l), CMPLX(1.l, 0.l)},
			{CMPLX(1.l, 0.l), CMPLX(0.l, 0.l)}
		},
		{
			{CMPLX(0.l, 0.l), CMPLX(0.l, -1.l)},
			{CMPLX(0.l, 1.l), CMPLX(0.l,  0.l)}
		},
		{
			{CMPLX(1.l, 0.l), CMPLX( 0.l, 0.l)},
			{CMPLX(0.l, 0.l), CMPLX(-1.l, 0.l)}
		}
	};

	print("sigma[0]", sigma[0]);
	print("sigma[1]", sigma[1]);
	print("sigma[2]", sigma[2]);

// complex 3d vector
	// only real values
	Vec< CMPLX >
	p = {CMPLX(3.l, 0.l), CMPLX(3.l, 0.l), CMPLX(5.l, 0.l)};

	print("p", p);

	Mat< CMPLX >
	sXp = {sigma[0] * p[0] + sigma[1] * p[1] + sigma[2] * p[2]};

	print("sXp", sXp);

// invert (sigma x p)
	// complex vector operator
	Mat< CMPLX >
	sXpi = inv(sXp);

	print("sXpi = inv(sXp)\nround(sXpi, 4)", round(sXpi, 4));
	print("round(sXpi | sXp, 4)", round(sXpi | sXp, 4));
	print("round(sXp | sXpi, 4)", round(sXp | sXpi, 4));
	
	return 0;
}

```
output:
```
sigma[0]
 (0,0) (1,0)
 (1,0) (0,0)

sigma[1]
  (0,0) (0,-1)
  (0,1)  (0,0)

sigma[2]
  (1,0)  (0,0)
  (0,0) (-1,0)

p
(3,0)  (3,0)  (5,0)  

sXp
  (5,0) (3,-3)
  (3,3) (-5,0)

sXpi = inv(sXp)
round(sXpi, 4)
       (0.1163,0) (0.0698,-0.0698)
  (0.0698,0.0698)      (-0.1163,0)

round(sXpi | sXp, 4)
 (1,0) (0,0)
 (0,0) (1,0)

round(sXp | sXpi, 4)
 (1,0) (0,0)
 (0,0) (1,0)
```

or calulate a checker board matrix from a vector
  
  
```
Vec<int>
v = {0,1,0,1,0,1,0,1};
v = 2 * v - 1;

Mat<int> checkerboard = ( 1 + ( v ^ -v ) ) / 2;

std::cout << "checkerboard:\n" << checkerboard << std::endl;
```
output:
```
checkerboard:
 0 1 0 1 0 1 0 1
 1 0 1 0 1 0 1 0
 0 1 0 1 0 1 0 1
 1 0 1 0 1 0 1 0
 0 1 0 1 0 1 0 1
 1 0 1 0 1 0 1 0
 0 1 0 1 0 1 0 1
 1 0 1 0 1 0 1 0
```
