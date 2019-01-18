//@start
#include "algebra.hpp"
#include "codeprinter.hpp"
#include <time.h>

typedef Vec< double > VD;
typedef Mat< double > MD;
typedef Tsr< double > TD;

//@some useful function definitions
// just for neuro
double
sigmoid( double const & p_net, double const & p_ymin = 0., double const & p_ymax = 1. ) {

	return p_ymin + ( p_ymax - p_ymin ) / ( 1. + exp( - p_net ) );
}

// activation function for of input values between 0 and 1
double
act_0p1( double const & p_net ) {

	return sigmoid( p_net, 0., 1. );
}

// activation function for of input values between -1 and +1
double
act_m1p1( double const & p_net ) {

	return sigmoid( p_net, -1., 1. );
}

// first derivative of sigmoid function
double
diffSigmoid( double const & p_y, double const & p_ymin = 0., double const & p_ymax = 1. ) {

	double
	a = ( p_y - p_ymin ) / ( p_ymax - p_ymin );

	return a - a * a;
}

// first derivative of activation function act_0p1
double
diffAct_0p1( double const & p_y ) {

	return diffSigmoid( p_y, 0., 1. );
}

// first derivative of activation function act_m1p1
double
diffAct_m1p1( double const & p_y ) {

	return diffSigmoid( p_y, -1., 1. );
}

// add a bias neuron of constant .9 to vector
VD
& addBias( VD & p_vec ) {

	p_vec.push_back( .9 );

	return p_vec;
}

int
main( ) {

	srand( time_t( nullptr ) );

	// arg isthis file
	CodePrinter
	codeprinter( "../AlgebraWithSTL/main.cpp" );

	codeprinter.print( "examples" );
//@examples
	// Examples for using algebra.hpp
	// -------- --- ----- -----------
	// next with [ENTER]
	// exit with x or X or q or Q and [ENTER] + [ENTER]
//@
	CodePrinter::WFE( );

	codeprinter.print( "create a vector" );
//@create a vector
	VD
	u = { 0., 1., 2., 3. },
	v = { 1., 2., 3., 5. };
//@
	CodePrinter::WFE( );

	codeprinter.print( "print a vector" );
//@print a vector
	std::cout << "u" << std::endl << u << std::endl << std::endl;
	// the same as before, just for lazyness
	print( "u", u );
//@
	CodePrinter::WFE( );

	codeprinter.print( "operator vector" );
//@operator vector
	print( "+u", +u );
	print( "-u", -u );
//@
	CodePrinter::WFE( );

	codeprinter.print( "vector operator scalar" );
//@vector operator scalar
	print( "u + .5", u + .5 );
	print( "u - .5", u - .5 );
	print( "u * .5", u * .5 );
	print( "u / .5", u / .5 );
//@
	CodePrinter::WFE( );

	codeprinter.print( "vector assignment operator scalar" );
//@vector assignment operator scalar
	print( "u += .5", u += .5 );
	print( "u -= .5", u -= .5 );
	print( "u *= .5", u *= .5 );
	print( "u /= .5", u /= .5 );
//@
	CodePrinter::WFE( );

	codeprinter.print( "scalar operator vector" );
//@scalar operator vector
	print( ".5 + u", .5 + u );
	print( ".5 - u", .5 - u );
	print( ".5 * u", .5 * u );
	print( ".5 / u", .5 / u );
//@
	CodePrinter::WFE( );

	codeprinter.print( "vector operator vector" );
//@vector operator vector
	print( "u + v", u + v );
	print( "u - v", u - v );
	print( "u * v", u * v );
	print( "u / v", u / v );
//@
	CodePrinter::WFE( );

	codeprinter.print( "vector assignment operator vector" );
//@vector assignment operator vector
	print( "u += v", u += v );
	print( "u -= v", u -= v );
	print( "u *= v", u *= v );
	print( "u /= v", u /= v );
//@
	CodePrinter::WFE( );

	codeprinter.print( "create a matrix" );
//@create a matrix
	MD
	a = {
		{ 1., 2., 3., 4. },
		{ -2., 0., 2., 0. } };
//@
	CodePrinter::WFE( );

	codeprinter.print( "operator matrix" );
//@operator matrix
	print( "a", a );
	print( "+a", +a );
	print( "-a", -a );

	// ~ gives the transposed matrix
	print( "~a", ~a );
//@
	CodePrinter::WFE( );

	codeprinter.print( "matrix operator scalar" );
//@matrix operator scalar
	print( "a + .5", a + .5 );
	print( "a - .5", a - .5 );
	print( "a * .5", a * .5 );
	print( "a / .5", a / .5 );
//@
	CodePrinter::WFE( );

	codeprinter.print( "matrix assignment operator scalar" );
//@matrix assignment operator scalar
	print( "a += .5", a += .5 );
	print( "a -= .5", a -= .5 );
	print( "a *= .5", a *= .5 );
	print( "a /= .5", a /= .5 );
//@
	CodePrinter::WFE( );

	codeprinter.print( "scalar operator matrix" );
//@scalar operator matrix
	print( ".5 + a", .5 + a );
	print( ".5 - a", .5 - a );
	print( ".5 * a", .5 * a );
	print( ".5 / a", .5 / a );
//@
	CodePrinter::WFE( );

	codeprinter.print( "matrix operator vector" );
//@matrix operator vector
	VD
	w = { -1., 1. };

	print( "a", a );
	print( "w", w );
	print( "a + w", a + w );
	print( "a - w", a - w );
	print( "a * w", a * w );
	print( "a / w", a / w );
//@
	CodePrinter::WFE( );

	codeprinter.print( "matrix assignment operator vector" );
//@matrix assignment operator vector
	print( "a += w", a += w );
	print( "a -= w", a -= w );
	print( "a *= w", a *= w );
	print( "a /= w", a /= w );
//@
	CodePrinter::WFE( );

	codeprinter.print( "vector operator matrix" );
//@vector operator matrix
	print( "u + a", u + a );
	print( "u - a", u - a );
	print( "u * a", u * a );
	print( "u / a", u / a );
//@
	CodePrinter::WFE( );

	codeprinter.print( "matrix operator matrix" );
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
//@
	CodePrinter::WFE( );

	codeprinter.print( "matrix assignment operator matrix" );
//@matrix assignment operator matrix
	print( "a += b", a += b );
	print( "a -= b", a -= b );
	print( "a *= b", a *= b );
	print( "a /= b", a /= b );
//@
	CodePrinter::WFE( );

	codeprinter.print( "some algebra" );
//@some algebra
	print( "u", u );
	print( "v", v );

//  scalar product
	print( "u | v", u | v );
//@
	CodePrinter::WFE( );

	codeprinter.print( "matrix times vector" );
//@matrix times vector
	print( "u", u );
	print( "b", b );
	print( "b | u", b | u );
//@
	CodePrinter::WFE( );

	codeprinter.print( "vector times matrix" );
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
	CodePrinter::WFE( );

	codeprinter.print( "matrix times matrix" );
//@matrix times matrix
	print( "a", a );
	print( "b", b );
	print( "a | ~b", a | ~b );
	print( "b | a", b | a );
	print( "b |= a", b |= a );
//@
	CodePrinter::WFE( );

	codeprinter.print( "outer product" );
//@outer product
	print( "u", u );
	print( "v", v );
	print( "u ^ v", u ^ v );
	print( "v ^ u", v ^ u );
//@
	CodePrinter::WFE( );

	codeprinter.print( "some functions" );
//@some functions
	a = {
		{ 3., 4. },
		{ -4., 3. } };

	a +=  .0001;

	print( "a", a );
	print( "~a", ~a );
	print( "inv( a )", inv( a ) );
	print( "abs( a )", det( a ) );
	print( "vcnst< double >( 2, 3 )", vcnst< double >( 2, 3 ) );
	print( "round( a / 10., 1 )", round( a / 10., 1 ) );
	print( "eye< double >( 5 )", eye< double >( 5 ) );
	print( "mcnst< double >( 2, 3, 6 )", mcnst< double >( 2, 3, 6. ) );
	print( "vrnd< double >( 3 )", vrnd< double >( 3 ) );
	print( "mrnd< double >( 3, 2 )", mrnd< double >( 3, 2 ) );
	print( "round( 100. * mrnd< double >( 7, 2 ) )", round( 100. * mrnd< double >( 7, 2 ) ) );
	print( "round( 100. * mrnd< double >( 3 ) )", round( 100. * mrnd< double >( 3 ) ) );
//@
	CodePrinter::WFE( );

	codeprinter.print( "now some neuro" );
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
	CodePrinter::WFE( );

	codeprinter.print( "some useful function definitions" );

	CodePrinter::WFE( );

	codeprinter.print( "some definitions" );
//@some definitions
	//now some useful neuro lines

	// build two random weights matrices with values between -1 and +1
	// first one connects the 8 + 1 = 9 input neurons to 3 hidden neurons
	// second one connects the 3 + 1 = 4 hidden neurons to 3 output neurons
	TD // actually this is not a tensor but only a Vec< MD > aka std::vector< std::vector< std::vector< double > > >
	weights = {
		round( 2. * mrnd< double >( 3, 9 ) - 1., 2 ),
		round( 2. * mrnd< double >( 3, 4 ) - 1., 2 )
	};

	// show weights for input and hidden layers
	print( "weights[ 0 ]", weights[ 0 ] );
	print( "weights[ 1 ]", weights[ 1 ] );

	// we need 7 more vectors
	// 3 ones to store values of activation of neurons for 3 layers (in hidden out)
	// 2 ones to store values of net sums for 2 layers  (in hidden)
	// 2 ones to store error vectors delta  (in hidden)
	MD // actually this is not a matrix but only a Vec< VD > aka std::vector< std::vector< double > >
	neuron( 3 ),
	net( 2 ),
	delta( 2 );

	// an 8 dimensional unit matrix stores 8 teacher vectors (00000001, 00000010, ..., 10000000)
	// remember: position of the "One" is what should be learned by the net
	MD
	teacherIn = eye< double >( 8 );

	// here the answer set
	// every row of this matrix should be remembered by its representant in teacherIn
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
	CodePrinter::WFE( );

	codeprinter.print( "now remember pattern 3" );
//@now remember pattern 3
	// let's try a first pattern
	std::size_t
	pattern = 3;

	// send teacher vector 3 to neurons in first layer
	neuron[ 0 ] = teacherIn[ pattern ];

	// add a 9th bias neuron
	addBias( neuron[ 0 ] );
	print( "neuron[ 0 ]", neuron[ 0 ] );

	// multiply weights of input layer with neurons of input layer to get net sums for input layer
	net[ 0 ] = weights[ 0 ] | neuron[ 0 ];
	print( "net[ 0 ]", net[ 0 ] );

	// transform net sum to activation of neurons in hidden layer
	neuron[ 1 ] = trnsfrm( net[ 0 ], act_0p1 );

	// again add an additional bias neuron to hidden layer
	addBias( neuron[ 1 ] );
	print( "neuron[ 1 ]", neuron[ 1 ] );

	// multiply weights of hidden layer with neurons of hidden layer to get net sums for hidden layer
	net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
	print( "net[ 1 ]", net[ 1 ] );

	// transform net sum to activation of neurons in output layer
	neuron[ 2 ] = trnsfrm( net[ 1 ], act_0p1 );
	print( "neuron[ 2 ]", neuron[ 2 ] );
//@
	CodePrinter::WFE( );

	codeprinter.print( "teach pattern 3 / get deltas" );
//@teach pattern 3 / get deltas
	// how good was it?
	VD
	err = neuron[ 2 ] - teacherOut[ pattern ];

	print( "err", err );

	// calc derivative of activation of output neurons
	VD
	diffAct = trnsfrm( neuron[ 2 ], diffAct_0p1 );
	print( "diffAct", diffAct );

	// errors of hidden layer
	delta[ 1 ] = diffAct * err;
	print( "delta[ 1 ]", delta[ 1 ] );

	// calculate changes in weights as product of error vector of hidden layer with weights of hidden layer
	VD
	dw = delta[ 1 ] | weights[ 1 ];
	print( "dw", dw );

	// calc derivative of activation of hidden neurons
	diffAct = trnsfrm( neuron[ 1 ], diffAct_0p1 );
	print( "diffAct", diffAct );

	// errors of input layer
	delta[ 0 ] = diffAct * dw;
	print( "delta[ 0 ]", delta[ 0 ] );
//@
	CodePrinter::WFE( );

	codeprinter.print( "teach pattern 3 / change weights" );
//@teach pattern 3 / change weights
	// teaching rate is set to .5
	double
	eta = .5;

	// now the net should learn what it does wrong in pattern 3
	// calc the change in weights for hidden layer as outer product of errors and outputs in hidden layer
	MD
	dW = delta[ 1 ] ^ neuron[ 1 ];

	// adjust weights with our half of our teaching rate .5 in hidden layer
	weights[ 1 ] -= .5 * eta * dW;

	print( "delta[ 1 ]", delta[ 1 ] );
	print( "neuron[ 1 ]", neuron[ 1 ] );
	print( "dW", dW );
	print( "weights[ 1 ]", weights[ 1 ] );

	// calc the change in weights for input layer as outer product of errors and outputs in input layer
	dW = delta[ 0 ] ^ neuron[ 0 ];

	// adjust weights with our teaching rate .5 in input layer
	weights[ 0 ] -= eta * dW;

	print( "delta[ 0 ]", delta[ 0 ] );
	print( "neuron[ 0 ]", neuron[ 0 ] );
	print( "dW", dW );
	print( "weights[ 0 ]", weights[ 0 ] );
//@
	CodePrinter::WFE( );

	codeprinter.print( "remember pattern 3 again" );
//@remember pattern 3 again
	neuron[ 0 ] = teacherIn[ pattern ];
	addBias( neuron[ 0 ] );
	print( "neuron[ 0 ]", neuron[ 0 ] );

	net[ 0 ] = weights[ 0 ] | neuron[ 0 ];
	print( "net[ 0 ]", net[ 0 ] );

	neuron[ 1 ] = trnsfrm( net[ 0 ], act_0p1 );
	addBias( neuron[ 1 ] );
	print( "neuron[ 1 ]", neuron[ 1 ] );

	net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
	print( "net[ 1 ]", net[ 1 ] );

	neuron[ 2 ] = trnsfrm( net[ 1 ], act_0p1 );
	print( "neuron[ 2 ]", neuron[ 2 ] );

	VD
	errNow = neuron[ 2 ] - teacherOut[ pattern ];

	print( "error before:           ", round( err, 4 ) );
	print( "error now:              ", round( errNow, 4 ) );
	print( "improvememt in percent: ", 100. * round( ( err - errNow ) / errNow, 4 ) );
//@
	CodePrinter::WFE( );

	codeprinter.print( "now 10000 loops" );
//@now 10000 loops
	for( std::size_t loop = 1; loop <= 10000; ++ loop ) {

		pattern = rand( ) & 0x7;

		neuron[ 0 ] = teacherIn[ pattern ];
		addBias( neuron[ 0 ] );

		net[ 0 ] = weights[ 0 ] | neuron[ 0 ];

		neuron[ 1 ] = trnsfrm( net[ 0 ], act_0p1 );
		addBias( neuron[ 1 ] );

		net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
		neuron[ 2 ] = trnsfrm( net[ 1 ], act_0p1 );

		dw = neuron[ 2 ] - teacherOut[ pattern ];

		if( ( loop == 1 ) || ( loop == 10 ) || ( loop == 1e2 ) || ( loop == 1e3 ) || ( loop == 1e4 ) || ( loop == 1e5 ) ) {

			print( "loop", loop );
			print( "sse:", dw | dw );
		}

		diffAct = trnsfrm( neuron[ 2 ], diffAct_0p1 );
		delta[ 1 ] = diffAct * dw;

		dw = delta[ 1 ] | weights[ 1 ];
		diffAct = trnsfrm( neuron[ 1 ], diffAct_0p1 );
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

		neuron[ 1 ] = trnsfrm( net[ 0 ], act_0p1 );
		addBias( neuron[ 1 ] );

		net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
		neuron[ 2 ] = trnsfrm( net[ 1 ], act_0p1 );

		print( "-------------------------------------------" );
		print( "in", sub( round( neuron[ 0 ], 2 ), 0, neuron[ 0 ].size( ) - 1 ) );
		print( "out", round( neuron[ 2 ], 2 ) );
	}
//@
	CodePrinter::WFE( );

	codeprinter.print( "save the weights" );
//@save the weights
	save( "weights", weights );
//@
	CodePrinter::WFE( );

	codeprinter.print( "load weights again" );
//@load weights again

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

	for( pattern = 0; pattern < 8; ++ pattern ) {

		neuron[ 0 ] = teacherIn[ pattern ];
		addBias( neuron[ 0 ] );

		net[ 0 ] = weights[ 0 ] | neuron[ 0 ];

		neuron[ 1 ] = trnsfrm( net[ 0 ], act_0p1 );
		addBias( neuron[ 1 ] );

		net[ 1 ] = weights[ 1 ] | neuron[ 1 ];
		neuron[ 2 ] = trnsfrm( net[ 1 ], act_0p1 );

		print( "-------------------------------------------" );
		print( "in", sub( neuron[ 0 ], 0, neuron[ 0 ].size( ) - 1 ) );
		print( "out", round( neuron[ 2 ], 2 ) );
	}
//@
	CodePrinter::WFE( );

	codeprinter.print( "XOR { 0, 1 } -> { 0, 1 }" );
//@XOR { 0, 1 } -> { 0, 1 }

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
		addBias( o[ 0 ] );

		n[ 0 ] = xorW[ 0 ] | o[ 0 ];
		o[ 1 ] = trnsfrm( n[ 0 ], act_0p1 );
		addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], act_0p1 );

		VD
		dw = o[ 2 ] - xorOut[ pattern ];

		if( ( loop == 1 ) || ( loop == 10 ) || ( loop == 1e2 ) || ( loop == 1e3 ) || ( loop == 1e4 ) || ( loop == 1e5 ) || ( loop == 1e6 ) ) {

			print( "loop", loop );
			print( "sse:", dw | dw );
		}

		diffAct = trnsfrm( o[ 2 ], diffAct_0p1 );
		d[ 1 ] = diffAct * dw;

		dw = d[ 1 ] | xorW[ 1 ];
		diffAct = trnsfrm( o[ 1 ], diffAct_0p1 );
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

		o[ 1 ] = trnsfrm( n[ 0 ], act_0p1 );
		addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], act_0p1 );

		print( "-------------------------------------------" );
		print( "in", sub( o[ 0 ], 0, o[ 0 ].size( ) - 1 ) );
		print( "out", round( o[ 2 ], 2 ) );
	}
//@
	CodePrinter::WFE( );

	codeprinter.print( "XOR { -1, +1 } > { -1, +1 }" );
//@XOR { -1, +1 } > { -1, +1 }

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
		2. * mrnd< double >( 1, 3 ) - 1.
	};

	for( std::size_t loop = 1; loop <= 100000; ++ loop ) {

		std::size_t
		pattern = rand( ) & 0x3;

		o[ 0 ] = xorIn[ pattern ];
		addBias( o[ 0 ] );

		n[ 0 ] = xorW[ 0 ] | o[ 0 ];

		// note: now activation function is act_m1p1 not act_0m1 as before
		o[ 1 ] = trnsfrm( n[ 0 ], act_m1p1 );
		addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], act_m1p1 );

		VD
		dw = o[ 2 ] - xorOut[ pattern ];

		if( ( loop == 1 ) || ( loop == 10 ) || ( loop == 1e2 ) || ( loop == 1e3 ) || ( loop == 1e4 ) || ( loop == 1e5 ) || ( loop == 1e6 ) ) {

			print( "loop", loop );
			print( "sse:", dw | dw );
		}

		diffAct = trnsfrm( o[ 2 ], diffAct_m1p1 );
		d[ 1 ] = diffAct * dw;

		dw = d[ 1 ] | xorW[ 1 ];
		diffAct = trnsfrm( o[ 1 ], diffAct_m1p1 );
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

		o[ 1 ] = trnsfrm( n[ 0 ], act_m1p1 );
		addBias( o[ 1 ] );

		n[ 1 ] = xorW[ 1 ] | o[ 1 ];
		o[ 2 ] = trnsfrm( n[ 1 ], act_m1p1 );

		print( "-------------------------------------------" );
		print( "in", sub( o[ 0 ], 0, o[ 0 ].size( ) - 1 ) );
		print( "out", round( o[ 2 ], 2 ) );
	}
//@
	CodePrinter::WFE( );

	codeprinter.print( "once again but shorter by using some vectors on stack" );
//@once again but shorter by using some vectors on stack

	xorW = {

		2. * mrnd< double >( 2, 3 ) - 1.,
		2. * mrnd< double >( 1, 3 ) - 1.
	};

	for( std::size_t loop = 1; loop <= 100000; ++ loop ) {

		std::size_t
		pattern = rand( ) & 0x3;

		addBias( o[ 0 ] = xorIn[ pattern ] );
		addBias( o[ 1 ] = trnsfrm( xorW[ 0 ] | o[ 0 ], act_m1p1 ) );
		o[ 2 ] = trnsfrm( xorW[ 1 ] | o[ 1 ], act_m1p1 );

		d[ 1 ] = trnsfrm( o[ 2 ], diffAct_m1p1 ) * ( o[ 2 ] - xorOut[ pattern ] );
		d[ 0 ] = trnsfrm( o[ 1 ], diffAct_m1p1 ) * ( d[ 1 ] | xorW[ 1 ] );

		xorW[ 1 ] -= .5 * eta * ( d[ 1 ] ^ o[ 1 ] );
		xorW[ 0 ] -=      eta * ( d[ 0 ] ^ o[ 0 ] );
	}

	for( pattern = 0; pattern < 4; ++ pattern ) {

		o[ 0 ] = xorIn[ pattern ];

		for( std::size_t i = 0; i < 2; ++ i ) {

			addBias( o[ i ] );

			o[ i + 1 ] = trnsfrm( xorW[ i ] | o[ i ], act_m1p1 );
		}

		print( "-------------------------------------------" );
		print( "in", sub( o[ 0 ], 0, o[ 0 ].size( ) - 1 ) );
		print( "out", round( o[ 2 ], 4 ) );
	}
//@
	CodePrinter::WFE( );

	codeprinter.print( "ende" );
//@ende
	// finally a really brief example of a dreaming net
	// task for the next brain is to remember the letter "A"
	// having only pure emptiness as first input
	// by memorizing its own memories
	teacherIn = {
		{ 0, 0, 0, 0, 0 },
		{ 0, 0, 1, 0, 0 },
		{ 0, 1, 0, 1, 0 },
		{ 1, 1, 0, 1, 1 },
		{ 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 1 } };

	// remember simply what has to come next
	teacherOut = {
		teacherIn[ 1 ],
		teacherIn[ 2 ],
		teacherIn[ 3 ],
		teacherIn[ 4 ],
		teacherIn[ 5 ],
		teacherIn[ 0 ] };

	// we need again 2 matrices of weights
	TD
	brain( 2 );

	// every maps 5+1 neurons to 5
	for( std::size_t i = 0; i < 2; ++ i )
		brain[ i ] = 2. * mrnd< double >( 5, 6 ) - 1.;

	// learn 10000 sets
	for( std::size_t loop = 1; loop <= 10000; ++ loop ) {

		std::size_t
		pattern = static_cast< std::size_t >( rand( ) ) % teacherIn.size( );

		addBias( o[ 0 ] = teacherIn[ pattern ] );
		addBias( o[ 1 ] = trnsfrm( brain[ 0 ] | o[ 0 ], act_0p1 ) );
		o[ 2 ] = trnsfrm( brain[ 1 ] | o[ 1 ], act_0p1 );

		VD
		dw = o[ 2 ] - teacherOut[ pattern ];

		if( ( loop == 1 ) || ( loop == 10 ) || ( loop == 1e2 ) || ( loop == 1e3 ) || ( loop == 1e4 ) || ( loop == 1e5 ) || ( loop == 1e6 ) ) {

			print( "loop", loop );
			print( "sse:", dw | dw );
		}

		d[ 1 ] = trnsfrm( o[ 2 ], diffAct_0p1 ) * dw;
		d[ 0 ] = trnsfrm( o[ 1 ], diffAct_0p1 ) * ( d[ 1 ] | brain[ 1 ] );

		brain[ 1 ] -= .5 * eta * ( d[ 1 ] ^ o[ 1 ] );
		brain[ 0 ] -=      eta * ( d[ 0 ] ^ o[ 0 ] );
	}
	// now the brain hopefully knows the letter A

	// this is everything that's needed for remembering a pattern
	#define REM for( std::size_t i = 0; i < 2; ++ i ) { addBias( o[ i ] ); o[ i + 1 ] = trnsfrm( brain[ i ] | o[ i ], act_0p1 ); }

	// show our brain an empty vector and hope it will remember the letter A
	o[ 0 ] = { 0, 0, 0, 0, 0 };

	// now remember some memories
	std::cout << "Aaaaahhhhh!:" << std::endl;

	for( std::size_t i = 0; i < 3 * teacherIn.size( ); ++ i ) {

		REM

		std::cout << round( o[ 2 ] ) << std::endl;

		//brain, think what you've remembered!
		o[ 0 ] = o[ 2 ];
	}
//@

	CodePrinter::WFE( );

	codeprinter.print( "And now for something completely different" );
//@And now for something completely different

	typedef std::complex< long double > CMPLX;
//@

	CodePrinter::WFE( );

	codeprinter.print( "Pauli Matrices" );
//@Pauli Matrices
	Tsr< CMPLX >
	sigma = {
		{
			{ CMPLX( 0.l, 0.l ), CMPLX( 1.l, 0.l ) },
			{ CMPLX( 1.l, 0.l ), CMPLX( 0.l, 0.l ) } },
		{
			{ CMPLX( 0.l, 0.l ), CMPLX( 0.l, -1.l ) },
			{ CMPLX( 0.l, 1.l ), CMPLX( 0.l,  0.l ) } },
		{
			{ CMPLX( 1.l, 0.l ), CMPLX(  0.l, 0.l ) },
			{ CMPLX( 0.l, 0.l ), CMPLX( -1.l, 0.l ) } } };

	print( "sigma[ 0 ]", sigma[ 0 ] );
	print( "sigma[ 1 ]", sigma[ 1 ] );
	print( "sigma[ 2 ]", sigma[ 2 ] );
//@

	CodePrinter::WFE( );

	codeprinter.print( "some real 3d vector" );
//@some real 3d vector
	Vec< CMPLX >
	p1 = { CMPLX( 2.l, 0.l ), CMPLX( 3.l, 0.l ), CMPLX( 5.l, 0.l ) };

	print( "p1", p1 );

	Mat< CMPLX >
	sigma1Xp1 = { sigma[ 0 ] * p1[ 0 ] + sigma[ 1 ] * p1[ 1 ] + sigma[ 2 ] * p1[ 2 ] };

	print( "sigma1Xp1", sigma1Xp1 );
//@

	CodePrinter::WFE( );

	codeprinter.print( "text" );
//@text

	Vec< STR >
	ps = { "a", "b", "c" };

	print( "ps", ps );

	Mat< STR >
	ms = { { "a00", "a01", "a02" }, { "a10", "a11", "a12" } };

	print( "ms", ms );

	print( "~ms", ~ms );

	print( "ms | ~ms", ms | ~ms );

	print( "~ms | ms", ~ms | ms );

	print( "ms | ps", ms | ps );

	print( "ps | ~ms", ps | ~ms );

//@

	return 0;
}
//@end
