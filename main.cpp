#include "mlp.hpp"
#include "codeprinter.hpp"

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


//@some useful function definitions
// just for neuro
D
sigmoid(D const & p_net, D const & p_ymin = 0., D const & p_ymax = 1.) {

	return p_ymin + (p_ymax - p_ymin) / (1. + exp(- p_net));
}

// activation function for input values between 0 and 1
D
act_0p1(D const & p_net) {

	return sigmoid(p_net, 0., 1.);
}

// activation function for input values between -1 and +1
D
act_m1p1(D const & p_net) {

	return sigmoid(p_net, -1., 1.);
}

// first derivative of sigmoid function
D
diffSigmoid(D const & p_act , D const & p_ymin = 0., D const & p_ymax = 1.) {

	return .0001 + (p_act - p_ymin) * (p_ymax - p_act) / (p_ymax - p_ymin);
}

// first derivative of activation function act_0p1
D
diffAct_0p1(D const & p_act) {

	return diffSigmoid(p_act, 0., 1.);
}

// first derivative of activation function act_m1p1
D
diffAct_m1p1(D const & p_act) {

	return diffSigmoid(p_act, -1., 1.);
}

// add a bias neuron of constant 1. to vector
VD
& addBias(VD & p_vec) {

	p_vec.push_back(1.);

	return p_vec;
}

// remove bias neuron
VD
& remBias(VD & p_vec) {

	p_vec.pop_back();

	return p_vec;
}
//@

int
main() {

	srand(time_t(nullptr));

	// arg is this file
	CodePrinter
	codeprinter("../AlgebraWithSTL/main.cpp", 4),
	codeprinter_alg("../AlgebraWithSTL/algebra.hpp", 4),
	codeprinter_mlp("../AlgebraWithSTL/mlp.hpp", 4);

	codeprinter.print("some typedefs for std::vector< T >");
	codeprinter_alg.print("even more abbr.");
	codeprinter.print("examples");
//@examples
	// Examples for using algebra.hpp
	// -------- --- ----- -----------
	// next with [ENTER]
	// exit with x or X or q or Q and [ENTER] + [ENTER]
//@
	CodePrinter::WFE();

	codeprinter.print("create a vector");
//@create a vector
	VD
	u = {0., 1., 2., 3.},
	v = {1., 2., 3., 5.};
//@
	CodePrinter::WFE();

	codeprinter.print("print a vector");
//@print a vector
	std::cout << "u" << std::endl << u << std::endl << std::endl;
	// the same as before, just for lazyness
	print("u", u);
//@
	CodePrinter::WFE();

	codeprinter.print("add a value at the end of a vector");
//@add a value at the end of a vector
	print("push_back(u, 4.)", push_back(u, 4.));
	print("push_front(u, -1.)", push_front(u, -1.));
	print("pop_back(u)", pop_back(u));
	print("+u", +u);
	print("pop_front(u)", pop_front(u));
	print("+u", +u);

//@
	CodePrinter::WFE();


	codeprinter.print("add and remove a vector before and at the end of a matrix");
//@add and remove a vector before and at the end of a matrix
	MD
	matrix({{1,2}, {3,5}});

	print("push_back(matrix, VD({7,11))", push_back(matrix, VD({7., 11.})));
	print("matrix = ~matrix", matrix = ~matrix);
	print("push_front(matrix, VD({7,11, 13))", push_front(matrix, VD({7,11, 13})));
	print("pop_back(matrix)", pop_back(matrix));
	print("matrix", matrix);
	print("pop_front(matrix)", pop_front(matrix));
	print("matrix", matrix);

//@
	CodePrinter::WFE();

	codeprinter.print("insert a vector into a matrix");
//@insert a vector into a matrix
	print("u", u);
	print("insert_val(u, 0, -1.)", insert_val(u, 0, -1.));
	print("insert_val(u, 4, 4.)", insert_val(u, 4, 4.));
	print("insert_vec(u, 0, {-4., -3., -1.})", insert_vec(u, 0, {-4., -3., -2.}));
	print("insert_vec(u, 4, {4., 5., 6.})", insert_vec(u, 4, {4., 5., 6.}));
	print("insert_val(u, 1, .5)", insert_val(u, 1, .5));
	print("insert_vec(u, 1, {.25, .5, .75})", insert_vec(u, 1, {.25, .5, .75}));
//@
	CodePrinter::WFE();

	codeprinter.print("insert and remove a matrix into a matrix");
//@insert a vector into a matrix
	print("matrix", matrix);
	print("insert_val(MD({{1,2,3}, {7,8,9}}), 1, {4,5,6})", insert_val(MD({{1,2,3}, {7,8,9}}), 1, {4,5,6}));
	print("insert_vec(MD({{1,2,3},{13,14,15}}), 1, {{4,5,6},{7,8,9},{10,11,12}})", insert_vec(MD({{1,2,3},{13,14,15}}), 1, {{4,5,6},{7,8,9},{10,11,12}}));
	print("remove(MD({{1,2,3}, {3,4,5}, {7,8,9}}), 0)", remove(MD({{1,2,3}, {4,5,6}, {7,8,9}}), 0));
	print("remove(MD({{1,2,3}, {3,4,5}, {7,8,9}}), 1)", remove(MD({{1,2,3}, {4,5,6}, {7,8,9}}), 1));
	print("remove(MD({{1,2,3}, {3,4,5}, {7,8,9}}), 2)", remove(MD({{1,2,3}, {4,5,6}, {7,8,9}}), 2));
	print("~remove(~MD({{1,2,3}, {4,5,6}, {7,8,9}}), 1)", ~remove(~MD({{1,2,3}, {4,5,6}, {7,8,9}}), 1));
	print("remove({0,1,2,3,4,5,6,7,8,9}, 1, 8)", remove({0,1,2,3,4,5,6,7,8,9}, 1, 8));
	print("remove({0,1,2,3,4,5,6,7,8,9}, 3, 4)", remove({0,1,2,3,4,5,6,7,8,9}, 3, 4));
	print("remove({0,1,2,3,4,5,6,7,8,9}, 0, 9)", remove({0,1,2,3,4,5,6,7,8,9}, 0, 9));
	print("remove({0,1,2,3,4,5,6,7,8,9}, 9, 1)", remove({0,1,2,3,4,5,6,7,8,9}, 9, 1));
//@
	CodePrinter::WFE();

	codeprinter.print("operator vector");
//@operator vector
	print("+u", +u);
	print("-u", -u);
//@
	CodePrinter::WFE();

	codeprinter.print("vector operator scalar");
//@vector operator scalar
	print("u + .5", u + .5);
	print("u - .5", u - .5);
	print("u * .5", u * .5);
	print("u / .5", u / .5);
//@
	CodePrinter::WFE();

	codeprinter.print("vector assignment operator scalar");
//@vector assignment operator scalar
	print("u += .5", u += .5);
	print("u -= .5", u -= .5);
	print("u *= .5", u *= .5);
	print("u /= .5", u /= .5);
//@
	CodePrinter::WFE();

	codeprinter.print("scalar operator vector");
//@scalar operator vector
	print(".5 + u", .5 + u);
	print(".5 - u", .5 - u);
	print(".5 * u", .5 * u);
	print(".5 / u", .5 / u);
//@
	CodePrinter::WFE();

	codeprinter.print("vector operator vector");
//@vector operator vector
	print("u + v", u + v);
	print("u - v", u - v);
	print("u * v", u * v);
	print("u / v", u / v);
//@
	CodePrinter::WFE();

	codeprinter.print("vector assignment operator vector");
//@vector assignment operator vector
	print("u += v", u += v);
	print("u -= v", u -= v);
	print("u *= v", u *= v);
	print("u /= v", u /= v);
//@
	CodePrinter::WFE();

	codeprinter.print("create a matrix");
//@create a matrix
	MD
	a = {
	{+1., +2., +3., +4.},
	{-2., +0., +2., +0.}};
//@
	CodePrinter::WFE();

	codeprinter.print("operator matrix");
//@operator matrix
	print("a", a);
	print("+a", +a);
	print("-a", -a);

	// ~ gives the transposed matrix
	print("~a", ~a);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix operator scalar");
//@matrix operator scalar
	print("a + .5", a + .5);
	print("a - .5", a - .5);
	print("a * .5", a * .5);
	print("a / .5", a / .5);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix assignment operator scalar");
//@matrix assignment operator scalar
	print("a += .5", a += .5);
	print("a -= .5", a -= .5);
	print("a *= .5", a *= .5);
	print("a /= .5", a /= .5);
//@
	CodePrinter::WFE();

	codeprinter.print("scalar operator matrix");
//@scalar operator matrix
	print(".5 + a", .5 + a);
	print(".5 - a", .5 - a);
	print(".5 * a", .5 * a);
	print(".5 / a", .5 / a);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix operator vector");
//@matrix operator vector
	VD
	w = {-1., 1.};

	print("a", a);
	print("w", w);
	print("a + w", a + w);
	print("a - w", a - w);
	print("a * w", a * w);
	print("a / w", a / w);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix assignment operator vector");
//@matrix assignment operator vector
	print("a += w", a += w);
	print("a -= w", a -= w);
	print("a *= w", a *= w);
	print("a /= w", a /= w);
//@
	CodePrinter::WFE();

	codeprinter.print("vector operator matrix");
//@vector operator matrix
	print("u + a", u + a);
	print("u - a", u - a);
	print("u * a", u * a);
	print("u / a", u / a);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix operator matrix");
//@matrix operator matrix
	MD
	b = {
	{+1., +1., +1., +1.},
	{-1., +1., -1., +1.}};

	print("a", a);
	print("b", b);
	print("a + b", a + b);
	print("a - b", a - b);
	print("a * b", a * b);
	print("a / b", a / b);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix assignment operator matrix");
//@matrix assignment operator matrix
	print("a += b", a += b);
	print("a -= b", a -= b);
	print("a *= b", a *= b);
	print("a /= b", a /= b);
//@
	CodePrinter::WFE();

	codeprinter.print("some algebra");
//@some algebra
	print("u", u);
	print("v", v);
//  scalar product
	print("u | v", u | v);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix times vector");
//@matrix times vector
	print("u", u);
	print("b", b);
	print("b | u", b | u);
//@
	CodePrinter::WFE();

	codeprinter.print("vector times matrix");
//@vector times matrix
	a = {
	{+1., +1., +1., +1.},
	{+1., -1., +1., -1.},
	{+1., +1., -1., -1.},
	{+1., -1., -1., +1.}};

	print("v", v);
	print("a", a);
	print("v | a", v | a);
	print("v |= a", v |= a);
//@
	CodePrinter::WFE();

	codeprinter.print("matrix times matrix");
//@matrix times matrix
	print("a", a);
	print("b", b);
	print("a | ~b", a | ~b);
	print("b | a", b | a);
	print("b |= a", b |= a);
//@
	CodePrinter::WFE();

	codeprinter.print("dyadic product");
//@dyadic product
	print("u", u);
	print("v", v);
	print("u ^ v", u ^ v);
	print("v ^ u", v ^ u);
//@
	CodePrinter::WFE();

	codeprinter.print("some functions");
//@some functions
	a = {
	{+3., +4.},
	{-4., +3.}};

	a += .0001;

	print("a", a);
	print("~a", ~a);
	print("inv(a)", inv(a));
	print("det(a)", det(a));
	print("vcnst< D >(2, 3)", vcnst< D >(2, 3));
	print("round(a / 10., 1)", round(a / 10., 1));
	print("eye< D >(5)", eye< D >(5));
	print("mcnst< D >(2, 3, 6)", mcnst< D >(2, 3, 6.));
	print("vrnd< D >(3)", vrnd< D >(3));
	print("mrnd< D >(3, 2)", mrnd< D >(3, 2));
	print("round(100. * mrnd< D >(7, 2))", round(100. * mrnd< D >(7, 2)));
	print("round(100. * mrnd< D >(3))", round(100. * mrnd< D >(3)));
//@
	CodePrinter::WFE();

	codeprinter.print("And now for something completely different");
//@And now for something completely different
	typedef std::complex< long double > CMPLX;
//@
	CodePrinter::WFE();

	codeprinter.print("Pauli Matrices");
//@Pauli Matrices
	// complex vector operator
	Tsr< CMPLX >
	sigma = {
	{
	{CMPLX(0.l, 0.l), CMPLX(1.l, 0.l)},
	{CMPLX(1.l, 0.l), CMPLX(0.l, 0.l)}},
	{
	{CMPLX(0.l, 0.l), CMPLX(0.l, -1.l)},
	{CMPLX(0.l, 1.l), CMPLX(0.l,  0.l)}},
	{
	{CMPLX(1.l, 0.l), CMPLX( 0.l, 0.l)},
	{CMPLX(0.l, 0.l), CMPLX(-1.l, 0.l)}}};

	print("sigma[0]", sigma[0]);
	print("sigma[1]", sigma[1]);
	print("sigma[2]", sigma[2]);
//@
	CodePrinter::WFE();

	codeprinter.print("complex 3d vector");
//@complex 3d vector
	// only real values
	Vec< CMPLX >
	p = {CMPLX(3.l, 0.l), CMPLX(3.l, 0.l), CMPLX(5.l, 0.l)};

	print("p1", p);

	Mat< CMPLX >
	sigma1Xp1 = {sigma[0] * p[0] + sigma[1] * p[1] + sigma[2] * p[2]};

	print("sigma1Xp1", sigma1Xp1);
//@
	CodePrinter::WFE();

	codeprinter.print("invert (sigma x p1)");
//@invert (sigma x p1)
	// complex vector operator
	Mat< CMPLX >
	sXpi = inv(sigma1Xp1);

	print("sXpi = inv(sigma1Xp1)\nround(sXpi, 4)", round(sXpi, 4));
	print("round(sXpi | sigma1Xp1, 4)", round(sXpi | sigma1Xp1, 4));
	print("round(sigma1Xp1 | sXpi, 4)", round(sigma1Xp1 | sXpi, 4));
//@
	CodePrinter::WFE();

	codeprinter.print("now a realy big matrix 100 x 100");
//@now a realy big matrix 100 x 100
	Mat< long double >
	big100x100  = mrnd< long double >(100, 100) - .5l;

	// print only a 30x30 frame
	print(
	"round(sub(big100x100, 0, 0, 30, 30), 2)",
	round(sub(big100x100, 0, 0, 30, 30), 2));
//@
	CodePrinter::WFE();

	codeprinter.print("calc its inverse");
//@calc its inverse
	Mat< long double >
	big100x100i = inv(big100x100);

	// print only a 30x30 frame again
	print(
	"round(sub(big100x100i, 0, 0, 30, 30), 2)",
	round(sub(big100x100i, 0, 0, 30, 30), 2));
//@
	CodePrinter::WFE();

	codeprinter.print("check the result");
//@check the result
	print(
	"round(sub(big100x100i | big100x100, 0, 0, 100, 100), 2)",
	round(sub(big100x100i | big100x100, 0, 0, 100, 100), 2));
//@
	CodePrinter::WFE();

	codeprinter.print("calculate the determinant of big100x100");
//@calculate the determinant of big100x100
	long double
	detBig = det(big100x100);
	print("det(big100x100)",  detBig);
//@
	CodePrinter::WFE();

	codeprinter.print("calculate the determinant of big100x100i");
//@calculate the determinant of big100x100i
	long double
	detBigi = det(big100x100i);
	print("det(big100x100)",  detBigi);
	print("det(big100x100i) * det(big100x100)", detBigi * detBig);
//@
	CodePrinter::WFE();

	codeprinter.print("now some text fun");
//@now some text fun
	Vec< STR >
	ps = {"x", "y", "z"};

	print("ps", ps);

	Mat< STR >
	ms = {{"a11", "a12", "a13"}, {"a21", "a22", "a23"}};

	print("ms", ms);

	print("~ms", ~ms);

	print("ms | ps", ms | ps);

	print("ps | ~ms", ps | ~ms);

	print("ms | ~ms", ms | ~ms);

	print("~ms | ms", ~ms | ms);
//@
	CodePrinter::WFE();





	codeprinter.print("now some neuro");
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
	CodePrinter::WFE();

	codeprinter.print("some useful function definitions");

	CodePrinter::WFE();

	codeprinter.print("let's build a simple multilayer perceptron");
//@let's build a simple multilayer perceptron
	//now some useful neuro lines

	// build two random weights matrices with values between -1 and +1
	// first one connects the 8 + 1 = 9 input neurons to 3 hidden neurons
	// second one connects the 3 + 1 = 4 hidden neurons to 3 output neurons
	TD // actually this is not a tensor but only a Vec< MD > aka std::vector< std::vector< std::vector< D > > >
	weights = {
	round(2. * mrnd< D >(3, 9) - 1., 2),
	round(2. * mrnd< D >(3, 4) - 1., 2)
	};

	// show weights for input and hidden layers
	print("weights[0]", weights[0]);
	print("weights[1]", weights[1]);

	// we need 7 more vectors
	// one to store the input vector
	// 3 ones to store values of activation of neurons for 3 layers (in hidden out)
	// 2 ones to store values of net sums for 2 layers  (in hidden)
	// 2 ones to store error vectors delta  (in hidden)
	MD // actually this is not a matrix but only a Vec< VD > aka std::vector< std::vector< D > >
	neuron(3),
	net(2),   // input neurons don't need net sums, so the index for net[0] are the net sum for the 1st hidden layer
	delta(2); // input neurons don't need deltas, so the index for delta[0] are the deltas for the 1st hidden layer

	// an 8 dimensional unit matrix stores 8 teacher vectors (00000001, 00000010, ..., 10000000)
	// remember: position of the "One" is what should be learned by the net
	MD
	teacherIn = eye< D >(8);

	// here the answer set
	// every row of this matrix should be remembered by its representant in teacherIn
	MD
	teacherOut = {
	{0., 0., 0.},
	{0., 0., 1.},
	{0., 1., 0.},
	{0., 1., 1.},
	{1., 0., 0.},
	{1., 0., 1.},
	{1., 1., 0.},
	{1., 1., 1.}};
//@
	CodePrinter::WFE();

	codeprinter.print("now remember pattern 3");
//@now remember pattern 3
	// let's try a first pattern
	UI
	pattern = 3;

	// send teacher vector 3 to neurons in first layer
	neuron[0] = teacherIn[pattern];

	// add a 9th bias neuron
	addBias(neuron[0]);
	print("neuron[0]", neuron[0]);

	// multiply weights of input layer with neurons of input layer to get net sums for input layer
	net[0] = weights[0] | neuron[0];
	print("net[0]", net[0]);

	// transform net sum to activation of neurons in hidden layer
	neuron[1] = trnsfrm(net[0], act_0p1);

	// again add an additional bias neuron to hidden layer
	addBias(neuron[1]);
	print("neuron[1]", neuron[1]);

	// multiply weights of hidden layer with neurons of hidden layer to get net sums for hidden layer
	net[1] = weights[1] | neuron[1];
	print("net[1]", net[1]);

	// transform net sum to activation of neurons in output layer
	neuron[2] = trnsfrm(net[1], act_0p1);
	print("neuron[2]", neuron[2]);
//@
	CodePrinter::WFE();

	codeprinter.print("teach pattern 3 / get deltas");
//@teach pattern 3 / get deltas
	// how good was it?
	VD
	err = neuron[2] - teacherOut[pattern];

	print("err", err);

	// calc derivative of activation of output neurons without bias neuron
	VD
	diffAct = trnsfrm(neuron[2], diffAct_0p1);
	print("diffAct", diffAct);

	// errors of output layer
	delta[1] = diffAct * err;
	print("delta[1]", delta[1]);

	// calculate changes in weights as product of error vector of output layer with weights of hidden layer
	VD
	dw = delta[1] | weights[1];
	print("dw", dw);

	// calc derivative of activation of hidden neurons
	diffAct = trnsfrm(remBias(neuron[1]), diffAct_0p1);
	print("diffAct", diffAct);

	// errors of hidden layer
	delta[0] = diffAct * dw;
	print("delta[0]", delta[0]);
//@
	CodePrinter::WFE();

	codeprinter.print("teach pattern 3 / change weights");
//@teach pattern 3 / change weights
	// teaching rate is set to .5
	D
	eta = .5;

	// now the net should learn what it does wrong in pattern 3
	// calc the change in weights for hidden layer as outer product of errors in output layer and outputs in hidden layer
	MD
	dW = delta[1] ^ addBias(neuron[1]);

	// adjust weights with our half of our teaching rate .5 in hidden layer
	weights[1] -= .5 * eta * dW;

	print("delta[1]", delta[1]);
	print("neuron[1]", neuron[1]);
	print("dW", dW);
	print("weights[1]", weights[1]);

	// calc the change in weights for input layer as outer product of errors in the hidden layer and outputs in input layer
	dW = delta[0] ^ neuron[0];

	// adjust weights with our teaching rate .5 in input layer
	weights[0] -= eta * dW;

	print("delta[0]", delta[0]);
	print("neuron[0]", neuron[0]);
	print("dW", dW);
	print("weights[0]", weights[0]);
//@
	CodePrinter::WFE();

	codeprinter.print("remember pattern 3 again");
//@remember pattern 3 again
	neuron[0] = teacherIn[pattern];
	addBias(neuron[0]);
	print("neuron[0]", neuron[0]);

	net[0] = weights[0] | neuron[0];
	print("net[0]", net[0]);

	neuron[1] = trnsfrm(net[0], act_0p1);
	addBias(neuron[1]);
	print("neuron[1]", neuron[1]);

	net[1] = weights[1] | neuron[1];
	print("net[1]", net[1]);

	neuron[2] = trnsfrm(net[1], act_0p1);
	print("neuron[2]", neuron[2]);

	VD
	errNow = neuron[2] - teacherOut[pattern];

	print("error before:           ", round(err, 4));
	print("error now:              ", round(errNow, 4));
	print("improvememt in percent: ", 100. * round((err - errNow) / errNow, 4));
//@
	CodePrinter::WFE();

	codeprinter.print("now 10000 loops");
//@now 10000 loops
	for(UI loop = 1; loop <= 10000; ++ loop) {

		pattern = rand() & 0x7;

		neuron[0] = teacherIn[pattern];
		addBias(neuron[0]);

		net[0] = weights[0] | neuron[0];

		neuron[1] = trnsfrm(net[0], act_0p1);
		addBias(neuron[1]);

		net[1] = weights[1] | neuron[1];
		neuron[2] = trnsfrm(net[1], act_0p1);

		dw = neuron[2] - teacherOut[pattern];

		if((loop == 1) || (loop == 10) || (loop == 1e2) || (loop == 1e3) || (loop == 1e4) || (loop == 1e5)) {

			print("loop", loop);
			print("sse:", dw | dw);
		}

		diffAct = trnsfrm(neuron[2], diffAct_0p1);
		delta[1] = diffAct * dw;

		dw = delta[1] | weights[1];
		diffAct = trnsfrm(remBias(neuron[1]), diffAct_0p1);
		delta[0] = diffAct * dw;

		dW = delta[1] ^ addBias(neuron[1]);
		weights[1] -= .5 * eta * dW;

		dW = delta[0] ^ neuron[0];
		weights[0] -= eta * dW;
	}

	for(pattern = 0; pattern < 8; ++ pattern) {

		neuron[0] = teacherIn[pattern];
		addBias(neuron[0]);

		net[0] = weights[0] | neuron[0];

		neuron[1] = trnsfrm(net[0], act_0p1);
		addBias(neuron[1]);

		net[1] = weights[1] | neuron[1];
		neuron[2] = trnsfrm(net[1], act_0p1);

		print("-------------------------------------------");
		print("in", sub(round(neuron[0], 2), 0, neuron[0].size() - 1));
		print("out", round(neuron[2], 2));
	}
//@
	CodePrinter::WFE();

	codeprinter.print("save the weights");
//@save the weights
	save("weights", weights);
//@
	CodePrinter::WFE();

	codeprinter.print("load weights again");
//@load weights again
	VD
	wv;

	load("weights", wv);

	print("first vector", wv);

	MD
	wm;

	load("weights", wm);

	print("first matrix", wm);

	load("weights", weights);

	print("complete weights tensor", weights);

	for(pattern = 0; pattern < 8; ++ pattern) {

		neuron[0] = teacherIn[pattern];
		addBias(neuron[0]);

		net[0] = weights[0] | neuron[0];

		neuron[1] = trnsfrm(net[0], act_0p1);
		addBias(neuron[1]);

		net[1] = weights[1] | neuron[1];
		neuron[2] = trnsfrm(net[1], act_0p1);

		print("-------------------------------------------");
		print("in", sub(neuron[0], 0, neuron[0].size() - 1));
		print("out", round(neuron[2], 2));
	}
//@
	CodePrinter::WFE();

	codeprinter.print("XOR {0, 1} -> {0, 1}");
//@XOR {0, 1} -> {0, 1}

	MD
	o(3),
	n(2),
	d(2),

	xorIn = {
	{0., 0.},
	{0., 1.},
	{1., 0.},
	{1., 1.}},

	xorOut = {
	{0.},
	{1.},
	{1.},
	{0.}};

	srand(2);

	TD
	xorW = {

	2. * mrnd< D >(2, 3) - 1.,
	2. * mrnd< D >(1, 3) - 1.
	};

	for(UI loop = 1; loop <= 100000; ++ loop) {

		UI
		pattern = rand() & 0x3;

		o[0] = xorIn[pattern];
		addBias(o[0]);

		n[0] = xorW[0] | o[0];
		o[1] = trnsfrm(n[0], act_0p1);
		addBias(o[1]);

		n[1] = xorW[1] | o[1];
		o[2] = trnsfrm(n[1], act_0p1);

		dw = o[2] - xorOut[pattern];

		if((loop == 1) || (loop == 10) || (loop == 1e2) || (loop == 1e3) || (loop == 1e4) || (loop == 1e5) || (loop == 1e6)) {

			print("loop", loop);
			print("sse:", dw | dw);
		}

		diffAct = trnsfrm(o[2], diffAct_0p1);
		d[1] = diffAct * dw;

		dw = d[1] | xorW[1];
		diffAct = trnsfrm(remBias(o[1]), diffAct_0p1);
		d[0] = diffAct * dw;

		dW = d[1] ^ addBias(o[1]);
		xorW[1] -= .5 * eta * dW;

		dW = d[0] ^ o[0];
		xorW[0] -= eta * dW;
	}

	for(pattern = 0; pattern < 4; ++ pattern) {

		o[0] = xorIn[pattern];
		addBias(o[0]);

		n[0] = xorW[0] | o[0];

		o[1] = trnsfrm(n[0], act_0p1);
		addBias(o[1]);

		n[1] = xorW[1] | o[1];
		o[2] = trnsfrm(n[1], act_0p1);

		print("-------------------------------------------");
		print("in", sub(o[0], 0, o[0].size() - 1));
		print("out", round(o[2], 2));
	}
//@
	CodePrinter::WFE();

	codeprinter.print("XOR {-1, +1} > {-1, +1}");
//@XOR {-1, +1} > {-1, +1}

	xorIn = {
	{-1., -1.},
	{-1., +1.},
	{+1., -1.},
	{+1., +1.}};

	xorOut = {
	{-1.},
	{+1.},
	{+1.},
	{-1.}};

	xorW = {

	2. * mrnd< D >(2, 3) - 1.,
	2. * mrnd< D >(1, 3) - 1.
	};

	for(UI loop = 1; loop <= 100000; ++ loop) {

		UI
		pattern = rand() & 0x3;

		o[0] = xorIn[pattern];
		addBias(o[0]);

		n[0] = xorW[0] | o[0];

		// note: now activation function is act_m1p1 not act_0m1 as before
		o[1] = trnsfrm(n[0], act_m1p1);
		addBias(o[1]);

		n[1] = xorW[1] | o[1];
		o[2] = trnsfrm(n[1], act_m1p1);

		VD
		dw = o[2] - xorOut[pattern];

		if((loop == 1) || (loop == 10) || (loop == 1e2) || (loop == 1e3) || (loop == 1e4) || (loop == 1e5) || (loop == 1e6)) {

			print("loop", loop);
			print("sse:", dw | dw);
		}

		diffAct = trnsfrm(o[2], diffAct_m1p1);
		d[1] = diffAct * dw;

		dw = d[1] | xorW[1];
		diffAct = trnsfrm(remBias(o[1]), diffAct_m1p1);
		d[0] = diffAct * dw;

		dW = d[1] ^ addBias(o[1]);
		xorW[1] -= .5 * eta * dW;

		dW = d[0] ^ o[0];
		xorW[0] -= eta * dW;
	}

	for(pattern = 0; pattern < 4; ++ pattern) {

		o[0] = xorIn[pattern];
		addBias(o[0]);

		n[0] = xorW[0] | o[0];

		o[1] = trnsfrm(n[0], act_m1p1);
		addBias(o[1]);

		n[1] = xorW[1] | o[1];
		o[2] = trnsfrm(n[1], act_m1p1);

		print("-------------------------------------------");
		print("in", sub(o[0], 0, o[0].size() - 1));
		print("out", round(o[2], 2));
	}
//@
	CodePrinter::WFE();

	codeprinter.print("once again but shorter by using some vectors on stack");
//@once again but shorter by using some vectors on stack

	xorW = {

	2. * mrnd< D >(2, 3) - 1.,
	2. * mrnd< D >(1, 3) - 1.
	};

	for(UI loop = 1; loop <= 100000; ++ loop) {

		UI
		pattern = rand() & 0x3;

		addBias(o[0] = xorIn[pattern]);
		addBias(o[1] = trnsfrm(xorW[0] | o[0], act_m1p1));
		o[2] = trnsfrm(xorW[1] | o[1], act_m1p1);

		d[1] = trnsfrm(o[2], diffAct_m1p1) * (o[2] - xorOut[pattern]);
		d[0] = trnsfrm(remBias(o[1]), diffAct_m1p1) * (d[1] | xorW[1]);

		xorW[1] -= .5 * eta * (d[1] ^ addBias(o[1]));
		xorW[0] -=      eta * (d[0] ^ o[0]);
	}

	for(pattern = 0; pattern < 4; ++ pattern) {

		o[0] = xorIn[pattern];

		for(UI i = 0; i < 2; ++ i) {

			addBias(o[i]);

			o[i + 1] = trnsfrm(xorW[i] | o[i], act_m1p1);
		}

		print("-------------------------------------------");
		print("in", sub(o[0], 0, o[0].size() - 1));
		print("out", round(o[2], 4));
	}
//@
	CodePrinter::WFE();

	codeprinter.print("Aaaa");
//@Aaaa
	// finally a really brief example of a dreaming net
	// task for the next brain is to remember the letter "A"
	// having only pure emptiness as first input
	// by memorizing its own memories
	teacherIn = {
	//              minds
	//    | conscious | subconscious |
	{0, 0, 0, 0, 0,    0, 0},
	{0, 0, 1, 0, 0,    0, 0},
	{0, 1, 0, 1, 0,    0, 0},
	{1, 0, 0, 0, 1,    0, 0},
	{1, 1, 1, 1, 1,    0, 0},
	{1, 0, 0, 0, 1,    0, 1},
	{1, 0, 0, 0, 1,    1, 0},
	{1, 0, 0, 0, 1,    1, 1}};

	// remember simply what has to come next
	teacherOut = {
	teacherIn[1],
	teacherIn[2],
	teacherIn[3],
	teacherIn[4],
	teacherIn[5],
	teacherIn[6],
	teacherIn[7],
	teacherIn[0]};

	// we need again 2 matrices of weights
	TD
	brain(2);

	// every maps 7+1 neurons to 8
	for(UI i = 0; i < len(brain); ++ i) {

		brain[i] = 2. * mrnd< D >(7, 8) - 1.;
	}

	// learn 100000 sets
	for(UI loop = 1; loop <= 100000; ++ loop) {

		UI
		pattern = static_cast< UI >(rand()) % teacherIn.size();

		addBias(o[0] = teacherIn[pattern]);
		addBias(o[1] = trnsfrm(brain[0] | o[0], act_0p1));
		o[2] = trnsfrm(brain[1] | o[1], act_0p1);

		VD
		dw = o[2] - teacherOut[pattern];

		if((loop == 1) || (loop == 10) || (loop == 1e2) || (loop == 1e3) || (loop == 1e4) || (loop == 1e5) || (loop == 1e6)) {

			print("loop", loop);
			print("sse:", dw | dw);
		}

		d[1] = trnsfrm(o[2], diffAct_0p1) * dw;
		d[0] = trnsfrm(remBias(o[1]), diffAct_0p1) * (d[1] | brain[1]);
		brain[1] -= .5 * eta * (d[1] ^ addBias(o[1]));
		brain[0] -=      eta * (d[0] ^ o[0]);
	}
	// now the brain hopefully knows the letter A

	// this is everything that's needed for remembering a pattern
#define REM for(UI i = 0; i < len(brain); ++ i) {addBias(o[i]); o[i + 1] = trnsfrm(brain[i] | o[i], act_0p1);}

	// show our brain an empty vector and hope it will remember the letter A
	o[0] = {0, 0, 0, 0, 0, 0, 0};

	// now remember some memories
	std::cout << "Aaaa!:" << std::endl;

	for(UI i = 0; i < 4 * teacherIn.size(); ++ i) {

		REM

		for(UI j = 0; j < 5; ++ j) {

			std::cout << (o[2][j] < .5 ? ' ' : 'A');
		}

		std::cout << std::endl;

		//brain, think what you've remembered!
		o[0] = o[2];
	}
//@
	CodePrinter::WFE();



	codeprinter.print("Time to put all we know in a class");
//@Time to put all we know in a class
	// make a class mlp
	// use it
//@
	CodePrinter::WFE();

	codeprinter_mlp.print("Class Multilayer Perceptron");

	CodePrinter::WFE();

	codeprinter.print("prepare new task: multiply two 3 bit numbers");
//@prepare new task: multiply two 3 bit numbers

	MD
	x_y = mcnst< D >(64, 6),        // 64 x (3 + 3) bits: y = y2 y1 y0   x = x2 x1 x0
	x_times_y = mcnst< D >(64, 6);  // 64 x 6 bits as result of x * y

	for (UI y = 0; y < 8; ++ y) {

		for (UI x = 0; x < 8; ++ x) {

			UI
			j = (y << 3) + x;

			for (UI i = 0; i < 6; ++ i) {

				x_y      [j][5 - i] = (j         & (1ul << i)) == (1ul << i);
				x_times_y[j][5 - i] = ((y * x) & (1ul << i)) == (1ul << i);
			}
		}
	}

	print("x_y", ~ x_y);
	print("x_times_y", ~ x_times_y);
	srand(1);

	MLP
	mlp({6, 8, 12, 8, 6}, .2, 0., 1., -1., +1., 1);

	UI
	loops = 160000,
	lloop = loops / 10,
	ll    = 0;

	for(UI loop = 1; loop <= loops; ++ loop) {

		UI
		pId = rand() & 0x3f;

		mlp.remember(x_y[pId]);

		mlp.teach(x_times_y[pId]);

//		if((loop == 1) || (loop == 10) || (loop == 1e2) || (loop == 1e3) || (loop == 1e4) || (loop == 1e5) || (loop == 1e6)) {
		if(++ ll == lloop) {

			ll = 0;

			print("loop", loop);
			print("sse:", mlp.rms());
			std::cout
			<< " --------------------------------------------- " << std::endl
			<< "| Bits:                     0   correct 0     |" << std::endl
			<< "| Bits:                     1 incorrect 0     |" << std::endl
			<< "| Bits:                     2 incorrect 1     |" << std::endl
			<< "| Bits:                     3   correct 1     |" << std::endl
			<< " ---------- ---------------------------------- " << std::endl
			<< "| Y        |  X        ==>  Z = X * Y         |" << std::endl
			<< "| 2  1  0  |  2  1  0       5  4  3  2  1  0  |" << std::endl
			<< " ----------+---------------------------------- " << std::endl;

			for(UI i = 0; i < 64; ++ i) {

				mlp.remember(x_y[i]);

				std::cout
				<< "| "
				<< sub(x_y[i], 0, 3)
				<< "|  "
				<< sub(x_y[i], 3, 3)
				<< "==>  "
				<< round((2. * mlp.output() + x_times_y[i]))
				<< "|"
				<< std::endl;
			}

			std::cout
			<< " ---------- ---------------------------------- " << std::endl << std::endl;
		}
	}
//@
	CodePrinter::WFE();

	return 0;
}
