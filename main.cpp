//@start
#include <algebra.hpp>
#include <codeprinter.hpp>

typedef Vec< double > VD;
typedef Mat< double > MD;
typedef Vec< MD >     TD;

// just for neuro
double
sigmoid( double const & p_net, double const & p_ymin = 0., double const & p_ymax = 1. ) {

	return p_ymin + ( p_ymax - p_ymin ) / ( 1. + exp( - p_net ) );
}

double
actXOR( double const & p_net ) {

	return sigmoid( p_net, 0., 1. );
}

double
actXOR2( double const & p_net ) {

	return sigmoid( p_net, -1., 1. );
}

VD
trnsfrm( VD const & p_net, double ( *foo )( double const & ) ) {

	VD
	r( p_net );

	fOr( r, assign, foo );

	return r;
}

double
diffSigmoid( double const & p_y, double const & p_ymin = 0., double const & p_ymax = 1. ) {

	double
	a = ( p_y - p_ymin ) / ( p_ymax - p_ymin );

	return a - a * a;
}

double
diffActXOR( double const & p_y ) {

	return diffSigmoid( p_y, 0., 1. );
}

double
diffActXOR2( double const & p_y ) {

	return diffSigmoid( p_y, -1., 1. );
}

VD
& addBias( VD & p_vec ) {

	p_vec.push_back( .9 );

	return p_vec;
}

bool
waitForENTER( ) {

	std::cout << "press [ENTER] for next step!\n";

	int
	c = std::cin.get( );

	return ( c == 'x' ) || ( c == 'X' ) || ( c == 'q' ) || ( c == 'Q' );
}

void
print( std::string const & p_string ) {

	std::cout << p_string << std::endl;
}

template< typename T >
void
print( std::string const & p_string, T const & p_v ) {

	print( p_string );

	std::cout << p_v << std::endl << std::endl;
}

#define WFE if( waitForENTER( ) ) return 0;

int
main( ) {

	// hier muss der Pfasd zum main.cpp bei dir auf dem rechner eingetragen sein
	CodePrinter
	fp( "../AlgebraWithSTL/main.cpp" );

	fp.print( "examples" );
//@examples
//	Examples for using algebra.hpp
//	-------- --- ----- -----------
//	next with [ENTER]
//  exit with x or X or q or Q and [ENTER] + [ENTER]
//@
	WFE

	fp.print( "create a vector" );
//@create a vector
	VD
	u = { 1., 2., 3., 4. };

	// print a vector
	std::cout << "u" << std::endl << u << std::endl << std::endl;

//	the same code as before, just for lazyness
	print( "u", u );
//@
	WFE

	fp.print( "operator vector" );
//@operator vector

//	some unary operators
	print( "u", u );
	print( "+u", +u );
	print( "-u", -u );
//@
	WFE

	fp.print( "vector op scalar" );
//@vector op scalar
	print( "u + .5", u + .5 );
	print( "u - .5", u - .5 );
	print( "u * .5", u * .5 );
	print( "u / .5", u / .5 );

	print( "u += .5", u += .5 );
	print( "u -= .5", u -= .5 );
	print( "u *= .5", u *= .5 );
	print( "u /= .5", u /= .5 );
//@
	WFE

	fp.print( "scalar operator vector" );
//@scalar operator vector
	print( ".5 + u", .5 + u );
	print( ".5 - u", .5 - u );
	print( ".5 * u", .5 * u );
	print( ".5 / u", .5 / u );
//@
	WFE

	fp.print( "vector operator vector" );
//@vector operator vector
	VD
	v = { 1., -1., 1., -1. };

	print( "u", u );
	print( "v", v );
	print( "u + v", u + v );
	print( "u - v", u - v );
	print( "u * v", u * v );
	print( "u / v", u / v );

	print( "u += v", u += v );
	print( "u -= v", u -= v );
	print( "u *= v", u *= v );
	print( "u /= v", u /= v );
//@
	WFE

	fp.print( "operator matrix" );
//@operator matrix
	MD
	a = {
		{ 1., 2., 3., 4. },
		{ -2., 0., 2., 0. } };

//	some unary operators
	print( "a", a );
	print( "+a", +a );
	print( "-a", -a );

//	~ gives the transposed matrix
	print( "~a", ~a );
//@
	WFE

	fp.print( "matrix operator scalar" );
//@matrix operator scalar
	print( "a + .5", a + .5 );
	print( "a - .5", a - .5 );
	print( "a * .5", a * .5 );
	print( "a / .5", a / .5 );
	print( "a += .5", a += .5 );
	print( "a -= .5", a -= .5 );
	print( "a *= .5", a *= .5 );
	print( "a /= .5", a /= .5 );
//@
	WFE

	fp.print( "scalar operator matrix" );
//@scalar operator matrix
	print( ".5 + a", .5 + a );
	print( ".5 - a", .5 - a );
	print( ".5 * a", .5 * a );
	print( ".5 / a", .5 / a );
//@
	WFE

	fp.print( "matrix operator vector" );
//@matrix operator vector
	u = { -1., 1. };

	print( "a", a );
	print( "u", u );
	print( "a + u", a + u );
	print( "a - u", a - u );
	print( "a * u", a * u );
	print( "a / u", a / u );
	print( "a += u", a += u );
	print( "a -= u", a -= u );
	print( "a *= u", a *= u );
	print( "a /= u", a /= u );
//@
	WFE

	fp.print( "vector operator matrix" );
//@vector operator matrix
	u = v;

	print( "a", a);
	print( "u", u );
	print( "u + a", u + a );
	print( "u - a", u - a );
	print( "u * a", u * a );
	print( "u / a", u / a );
//@
	WFE

	fp.print( "matrix operator matrix" );
//@matrix operator matrix
	MD
	b = {
		{ 1., 1., 1., 1. },
		{ -1., 1., -1., 1. } };

	print( "a", a );
	print( "b", b );
	print( "a + b", a + b );
	print( "a - b", a - b );
	print( "a * b", a * b );
	print( "a / b", a / b );
	print( "a += b", a += b );
	print( "a -= b", a -= b );
	print( "a *= b", a *= b );
	print( "a /= b", a /= b );
//@
	WFE

	fp.print( "some algebra" );
//@some algebra
	print( "u", u );
	print( "v", v );
//  scalar product
	print( "u | v", u | v );
//@
	WFE

	fp.print( "matrix times vector" );
//@matrix times vector
	print( "u", u );
	print( "b", b );
	print( "b | u", b | u );
//@
	WFE

	fp.print( "vector times matrix" );
//@vector times matrix
	a = {
		{ 1., 1., 1., 1. },
		{ 1., -1., 1., -1. },
		{ 1., 1., -1., -1. },
		{ 1., -1., -1., 1. } };

	print( "v", v );
	print( "a", a );
	print( "v | a", v | a );
	print( "v |= a", v |= a );
//@
	WFE

	fp.print( "matrix times matrix" );
//@matrix times matrix
	print( "a", a );
	print( "b", b );
	print( "a | ~b", a | ~b );
	print( "b | a", b | a );
	print( "b |= a", b |= a );
//@
	WFE

	fp.print( "outer product" );
//@outer product
	u = { 1., 2. };
	v = { -1., 0., 1. };

	print( "u", u );
	print( "v", v );
	print( "u ^ v", u ^ v );
	print( "v ^ u", v ^ u );
//@
	WFE

	fp.print( "some functions" );
//@some functions
	a = {
		{ 3., 4. },
		{ -4., 3. } };

	a +=  .0001;

	print( "a", a );
	print( "~a", ~a );
	print( "inv( a )", inv( a ) );
	print( "abs( a )", abs( a ) );
	print( "round( a / 10., 1 )", round( a / 10., 1 ) );
	print( "eye< double >( 5 )", eye< double >( 5 ) );
	print( "ones< double >( 2 )", ones< double >( 2 ) );
	print( "ones< double >( 2, 3 )", ones< double >( 2, 3 ) );
	print( "vrnd< double >( 3 )", vrnd< double >( 3 ) );
	print( "round( 100. * mrnd< double >( 7, 2 ) )", round( 100. * mrnd< double >( 7, 2 ) ) );
	print( "round( 100. * mrnd< double >( 3 ) )", round( 100. * mrnd< double >( 3 ) ) );
//@
	WFE

	fp.print( "now some neuro" );
//@now some neuro
	// goal is to create a multi layer perceptron
	// that numbers the 8 possible positions of the only one in a vector of zeros
	// 10000000 => 000
	// 00010000 => 011
	// 00000001 => 111

	// mlp:
	// inputs 8 neurons + 1 bias
	// hidden 3 neurons + 1 bias
	// output 3 neurons
//@
	WFE

	fp.print( "some definitions" );
//@some definitions
//and finally some usefull lines for neuro
	TD
	weights = {
		round( 2. * mrnd< double >( 3, 9 ) - 1., 2 ),
		round( 2. * mrnd< double >( 3, 4 ) - 1., 2 )
	};

	print( "weights[ 0 ]", weights[ 0 ] );
	print( "weights[ 1 ]", weights[ 1 ] );

	MD
	neuron( 3 ),
	net( 2 ),
	delta( 2 );

	MD
	teacherIn = eye< double >( 8 );

	MD
	teacherOut = {
		{ 0., 0., 0. },
		{ 0., 0., 1. },
		{ 0., 1., 0. },
		{ 0., 1., 1. },
		{ 1., 0., 0. },
		{ 1., 0., 1. },
		{ 1., 1., 0. },
		{ 1., 1., 1. } };
//@
	WFE

	std::size_t
	pattern = 3;

	fp.print( "now remember pattern 3" );
//@now remember pattern 3
	neuron[ 0 ] = teacherIn[ pattern ];
	addBias( neuron[ 0 ] );
	print( "neuron[ 0 ]", neuron[ 0 ] );

	net[ 0 ] = weights[ 0 ] | neuron[ 0 ];
	print( "net[ 0 ]", net[ 0 ] );

	neuron[ 1 ] = trnsfrm( net[ 0 ], actXOR );
	addBias( neuron[ 1 ] );
	print( "neuron[ 1 ]", neuron[ 1 ] );

	net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
	print( "net[ 1 ]", net[ 1 ] );

	neuron[ 2 ] = trnsfrm( net[ 1 ], actXOR );
	print( "neuron[ 2 ]", neuron[ 2 ] );
//@
	WFE

	fp.print( "teach pattern 3 / get deltas" );
//@teach pattern 3 / get deltas
	VD
	err = neuron[ 2 ] - teacherOut[ pattern ];

	print( "err", err );

	VD
	diffAct = trnsfrm( neuron[ 2 ], diffActXOR );
	print( "diffAct", diffAct );

	delta[ 1 ] = diffAct * err;
	print( "delta[ 1 ]", delta[ 1 ] );

	VD
	dw = delta[ 1 ] | weights[ 1 ];
	print( "dw", dw );

	diffAct = trnsfrm( neuron[ 1 ], diffActXOR );
	print( "diffAct", diffAct );

	delta[ 0 ] = diffAct * dw;
	print( "delta[ 0 ]", delta[ 0 ] );
//@
	WFE

	fp.print( "teach pattern 3 / change weights" );
//@teach pattern 3 / change weights
	double
	eta = .5;

	MD
	dW = delta[ 1 ] ^ neuron[ 1 ];

	weights[ 1 ] -= .5 * eta * dW;

	print( "delta[ 1 ]", delta[ 1 ] );
	print( "neuron[ 1 ]", neuron[ 1 ] );
	print( "dW", dW );
	print( "weights[ 1 ]", weights[ 1 ] );

	dW = delta[ 0 ] ^ neuron[ 0 ];

	weights[ 0 ] -= eta * dW;

	print( "delta[ 0 ]", delta[ 0 ] );
	print( "neuron[ 0 ]", neuron[ 0 ] );
	print( "dW", dW );
	print( "weights[ 0 ]", weights[ 0 ] );
//@
	WFE

	fp.print( "remember pattern 3 again" );
//@remember pattern 3 again
	neuron[ 0 ] = teacherIn[ pattern ];
	addBias( neuron[ 0 ] );
	print( "neuron[ 0 ]", neuron[ 0 ] );

	net[ 0 ] = weights[ 0 ] | neuron[ 0 ];
	print( "net[ 0 ]", net[ 0 ] );

	neuron[ 1 ] = trnsfrm( net[ 0 ], actXOR );
	addBias( neuron[ 1 ] );
	print( "neuron[ 1 ]", neuron[ 1 ] );

	net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
	print( "net[ 1 ]", net[ 1 ] );

	neuron[ 2 ] = trnsfrm( net[ 1 ], actXOR );
	print( "neuron[ 2 ]", neuron[ 2 ] );

	err = neuron[ 2 ] - teacherOut[ pattern ];

	print( "err", err );
//@
	WFE

	fp.print( "now 10000 loops" );
//@now 10000 loops
	for( std::size_t loop = 1; loop <= 10000; ++ loop ) {

		pattern = rand( ) & 0x7;

		neuron[ 0 ] = teacherIn[ pattern ];
		addBias( neuron[ 0 ] );

		net[ 0 ] = weights[ 0 ] | neuron[ 0 ];

		neuron[ 1 ] = trnsfrm( net[ 0 ], actXOR );
		addBias( neuron[ 1 ] );

		net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
		neuron[ 2 ] = trnsfrm( net[ 1 ], actXOR );

		dw = neuron[ 2 ] - teacherOut[ pattern ];

		if( ( loop == 1 ) || ( loop == 10 ) || ( loop == 1e2 ) || ( loop == 1e3 ) || ( loop == 1e4 ) || ( loop == 1e5 ) ) {

			print( "loop", loop );
			print( "sse:", dw | dw );
		}

		diffAct = trnsfrm( neuron[ 2 ], diffActXOR );
		delta[ 1 ] = diffAct * dw;

		dw = delta[ 1 ] | weights[ 1 ];
		diffAct = trnsfrm( neuron[ 1 ], diffActXOR );
		delta[ 0 ] = diffAct * dw;

		dW = delta[ 1 ] ^ neuron[ 1 ];
		weights[ 1 ] -= .5 * eta * dW;

		dW = delta[ 0 ] ^ neuron[ 0 ];
		weights[ 0 ] -= eta * dW;
	}

	for( pattern = 0; pattern < 8; ++ pattern ) {

		neuron[ 0 ] = teacherIn[ pattern ];
		addBias( neuron[ 0 ] );

		net[ 0 ] = weights[ 0 ] | neuron[ 0 ];

		neuron[ 1 ] = trnsfrm( net[ 0 ], actXOR );
		addBias( neuron[ 1 ] );

		net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
		neuron[ 2 ] = trnsfrm( net[ 1 ], actXOR );

		print( "-------------------------------------------" );
		print( "neuron[ 0 ]", round( neuron[ 0 ], 2 ) );
		print( "neuron[ 2 ]", round( neuron[ 2 ], 2 ) );
	}
//@
	WFE

	fp.print( "save the weights" );
//@save the weights
	save( "weights", weights );
//@
	WFE

	fp.print( "load the weights" );
//@load the weights

	VD
	wv;

	load( "weights", wv );

	print( "first vector", wv );

	MD
	wm;

	load( "weights", wm );

	print( "first matrix", wm );

	load( "weights", weights );

	print( "complete weights tensor", weights );
//@
	WFE

	fp.print( "XOR with bias" );
//@XOR with bias

	MD

	o( 3 ),
	n( 2 ),
	d( 2 ),

	xorIn = {
		{ 0., 0. },
		{ 0., 1. },
		{ 1., 0. },
		{ 1., 1. } },

	xorOut = {
		{ 0. },
		{ 1. },
		{ 1. },
		{ 0. } };

	TD
	xorW = {

		2. * mrnd< double >( 2, 3 ) - 1.,
		2. * mrnd< double >( 1, 3 ) - 1.
	};

	for( std::size_t loop = 1; loop <= 100000; ++ loop ) {

		std::size_t
		pattern = rand( ) & 0x3;

		o[ 0 ] = xorIn[ pattern ];
		// one needs the bias neuron
		// tried to skip it and remove respective weights
		// does not work
		addBias( o[ 0 ] );

		n[ 0 ] = xorW[ 0 ] | o[ 0 ];
		o[ 1 ] = trnsfrm( n[ 0 ], actXOR );
		// one needs the bias neuron
		// tried to skip it and remove respective weights
		// does not work
		addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], actXOR );

		VD
		dw = o[ 2 ] - xorOut[ pattern ];

		if( ( loop == 1 ) || ( loop == 10 ) || ( loop == 1e2 ) || ( loop == 1e3 ) || ( loop == 1e4 ) || ( loop == 1e5 ) ) {

			print( "loop", loop );
			print( "sse:", dw | dw );
		}

		diffAct = trnsfrm( o[ 2 ], diffActXOR );
		d[ 1 ] = diffAct * dw;

		dw = d[ 1 ] | xorW[ 1 ];
		diffAct = trnsfrm( o[ 1 ], diffActXOR );
		d[ 0 ] = diffAct * dw;

		dW = d[ 1 ] ^ o[ 1 ];
		xorW[ 1 ] -= .5 * eta * dW;

		dW = d[ 0 ] ^ o[ 0 ];
		xorW[ 0 ] -= eta * dW;
	}

	for( pattern = 0; pattern < 4; ++ pattern ) {

		o[ 0 ] = xorIn[ pattern ];
		addBias( o[ 0 ] );

		n[ 0 ] = xorW[ 0 ] | o[ 0 ];

		o[ 1 ] = trnsfrm( n[ 0 ], actXOR );
		addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], actXOR );

		print( "-------------------------------------------" );
		print( "neuron[ 0 ]", round( o[ 0 ], 2 ) );
		print( "neuron[ 2 ]", round( o[ 2 ], 2 ) );
	}
//@
	WFE

	fp.print( "XOR without 1 bias" );
//@XOR without 1 bias

	xorIn = {
		{ -1., -1. },
		{ -1., +1. },
		{ +1., -1. },
		{ +1., +1. } };

	xorOut = {
		{ -1. },
		{ +1. },
		{ +1. },
		{ -1. } };

	xorW = {

		2. * mrnd< double >( 2, 3 ) - 1.,
		2. * mrnd< double >( 1, 2 ) - 1.
	};

	for( std::size_t loop = 1; loop <= 100000; ++ loop ) {

		std::size_t
		pattern = rand( ) & 0x3;

		o[ 0 ] = xorIn[ pattern ];
		// one needs the bias neuron
		// tried to skip it and remove respective weights
		// does not work
		addBias( o[ 0 ] );

		n[ 0 ] = xorW[ 0 ] | o[ 0 ];
		o[ 1 ] = trnsfrm( n[ 0 ], actXOR2 );
		// one needs the bias neuron
		// tried to skip it and remove respective weights
		// does not work
		//addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], actXOR2 );

		VD
		dw = o[ 2 ] - xorOut[ pattern ];

		if( ( loop == 1 ) || ( loop == 10 ) || ( loop == 1e2 ) || ( loop == 1e3 ) || ( loop == 1e4 ) || ( loop == 1e5 ) ) {

			print( "loop", loop );
			print( "sse:", dw | dw );
		}

		diffAct = trnsfrm( o[ 2 ], diffActXOR2 );
		d[ 1 ] = diffAct * dw;

		dw = d[ 1 ] | xorW[ 1 ];
		diffAct = trnsfrm( o[ 1 ], diffActXOR2 );
		d[ 0 ] = diffAct * dw;

		dW = d[ 1 ] ^ o[ 1 ];
		xorW[ 1 ] -= .5 * eta * dW;

		dW = d[ 0 ] ^ o[ 0 ];
		xorW[ 0 ] -= eta * dW;
	}

	for( pattern = 0; pattern < 4; ++ pattern ) {

		o[ 0 ] = xorIn[ pattern ];
		addBias( o[ 0 ] );

		n[ 0 ] = xorW[ 0 ] | o[ 0 ];
		o[ 1 ] = trnsfrm( n[ 0 ], actXOR2 );
		//addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], actXOR2 );

		print( "-------------------------------------------" );
		print( "neuron[ 0 ]", round( o[ 0 ], 4 ) );
		print( "neuron[ 2 ]", round( o[ 2 ], 4 ) );
	}

	print( "xorWeights", xorW );
	save( "xorWeights", xorW );
//@
	WFE

	fp.print( "XOR from file" );
//@XOR from file
	//delete mind
	xorW = {{{}}};

	//load mind again

	load( "xorWeights", xorW );

	for( pattern = 0; pattern < 4; ++ pattern ) {

		o[ 0 ] = xorIn[ pattern ];
		addBias( o[ 0 ] );

		for( std::size_t l = 0; l < 2; ++ l ) {

			n[ l ] = xorW[ l ] | o[ l ];
			o[ l + 1 ] = trnsfrm( n[ l ], actXOR2 );
		//addBias( o[ 1 ] );
		}
		print( "-------------------------------------------" );
		print( "neuron[ 0 ]", round( o[ 0 ], 4 ) );
		print( "neuron[ 2 ]", round( o[ 2 ], 4 ) );
	}
//@
	return 0;
}
//@end
