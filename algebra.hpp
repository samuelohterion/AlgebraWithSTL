#ifndef ALGEBRA_HPP
#define ALGEBRA_HPP
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>
#include <string>
#include <string.h>
#include <sstream>
#include <fstream>
#include <cstring>
#include <initializer_list>

template<typename T > using
Vec = std::vector< T >;

template<typename T > using
Mat = std::vector< std::vector< T > >;

template<typename T >
std::size_t
n( Vec< T > const & p_vec ) {

	return p_vec.size( );
}

template<typename T >
std::size_t
nrows( Mat< T > const & p_mat ) {

	return p_mat.size( );
}

template<typename T >
std::size_t
ncols( Mat< T > const & p_mat ) {

	return p_mat[ 0 ].size( );
}

template < typename T >
void
assign( T & p_el, T const & p_val ) {

	p_el = p_val;
}

template < typename T >
void
cumsum( T & p_el, T const & p_val ) {

	p_el = p_el + p_val;
}

template < typename T >
void
cumprod( T & p_el, T const & p_val ) {

	p_el = p_el * p_val;
}

template < typename T >
Vec< T >
& fOr( Vec< T > & p_vec, void ( *acc )( T &, T const & ), T ( *foo )( T const & ) ) {

	for( auto & i : p_vec )

		acc( i, foo( i ) );

	return p_vec;
}

template < typename T >
Vec< T >
fOr( Vec< T > & p_lhs, Vec< T > & p_rhs, void ( * acc )( T &, T const & ), T ( *foo )( T const &, T const & ) ) {

	Vec< T >
	ret( n( p_lhs ) );

	auto
	el = p_lhs.cend( ),
	sl = p_lhs.cbegin( ),
	sr = p_rhs.cbegin( );

	auto
	d = ret.begin( );

	while( sl != el ) {

		acc( *d, foo( *sl, *sr ) );
	}

	return ret;
}

#define BINOPVV( OP, foo ) \
template< typename T > Vec< T > operator OP( Vec< T > const & p_lhs, Vec< T > const & p_rhs ) { Vec< T > ret( p_lhs.size( ) ); \
std::transform( p_lhs.cbegin( ), p_lhs.cend( ), p_rhs.cbegin( ), ret.begin( ), foo( ) ); return ret; }

BINOPVV( +, std::plus< T > )
BINOPVV( -, std::minus< T > )
BINOPVV( *, std::multiplies< T > )
BINOPVV( /, std::divides< T > )

#define BINASSGNOPVV( OP ) \
template< typename T > Vec< T > & operator OP( Vec< T > & p_lhs, Vec< T > const & p_rhs ) { \
auto il = p_lhs.begin( ), el = p_lhs.end( ); auto ir = p_rhs.cbegin( ); while( il != el ) { *il OP *ir; ++il; ++ir; } return p_lhs; }

BINASSGNOPVV( += )
BINASSGNOPVV( -= )
BINASSGNOPVV( *= )
BINASSGNOPVV( /= )

#define BINOPSV( OP ) \
template< typename T > Vec< T > operator OP( T const & p_lhs, Vec< T > p_rhs ) { for( T & v : p_rhs ) v = p_lhs OP v; return p_rhs; }

BINOPSV( + )
BINOPSV( - )
BINOPSV( * )
BINOPSV( / )

#define BINOPVS( OP ) \
template< typename T > Vec< T > operator OP( Vec< T > p_lhs, T const & p_rhs ) { for( T & v : p_lhs ) v = v OP p_rhs; return p_lhs; }

BINOPVS( + )
BINOPVS( - )
BINOPVS( * )
BINOPVS( / )

#define BINASSGNOPVS( OP ) \
template< typename T > Vec< T > & operator OP( Vec< T > & p_lhs, T const & p_rhs ) { for( T & v : p_lhs ) v OP p_rhs; return p_lhs; }

BINASSGNOPVS( += )
BINASSGNOPVS( -= )
BINASSGNOPVS( *= )
BINASSGNOPVS( /= )

#define BINOPMM( OP ) \
template< typename T > Mat< T > operator OP( Mat< T > const & p_lhs, Mat< T > const & p_rhs ) { \
Mat< T > ret( p_lhs.size( ) ); for( std::size_t i = 0; i < ret.size( ); ++ i ) ret[ i ] = p_lhs[ i ] OP p_rhs[ i ];	return ret; }

BINOPMM( + )
BINOPMM( - )
BINOPMM( * )
BINOPMM( / )

#define BINASSGNOPMM( OP ) \
template< typename T > Mat< T > & operator OP( Mat< T > & p_lhs, Mat< T > const & p_rhs ) { \
for( std::size_t i = 0; i < p_lhs.size( ); ++ i ) p_lhs[ i ] OP p_rhs[ i ]; return p_lhs; }

BINASSGNOPMM( += )
BINASSGNOPMM( -= )
BINASSGNOPMM( *= )
BINASSGNOPMM( /= )

#define BINOPSM( OP ) \
template< typename T > Mat< T > operator OP( T const & p_lhs, Mat< T > const & p_rhs ) { Mat< T > ret( p_rhs.size( ) ); \
for( std::size_t i = 0; i < ret.size( ); ++ i ) ret[ i ] = p_lhs OP p_rhs[ i ]; return ret; }

BINOPSM( + )
BINOPSM( - )
BINOPSM( * )
BINOPSM( / )

#define BINOPMS( OP ) \
template< typename T > Mat< T > operator OP( Mat< T > const & p_lhs, T const & p_rhs ) { Mat< T > ret( p_lhs.size( ) ); \
for( std::size_t i = 0; i < p_lhs.size( ); ++ i ) ret[ i ] = p_lhs[ i ] OP p_rhs; return ret; }

BINOPMS( + )
BINOPMS( - )
BINOPMS( * )
BINOPMS( / )

#define BINASSGNOPMS( OP ) \
template< typename T > Mat< T > & operator OP( Mat< T > & p_lhs, T const & p_rhs ) { \
for( auto & i : p_lhs ) i OP p_rhs; return p_lhs; }

BINASSGNOPMS( += )
BINASSGNOPMS( -= )
BINASSGNOPMS( *= )
BINASSGNOPMS( /= )

#define BINOPVM( OP ) \
template< typename T > Mat< T > operator OP( Vec< T > const & p_lhs, Mat< T > const & p_rhs ) { Mat< T > ret( p_rhs.size( ) ); \
for( std::size_t i = 0; i < ret.size( ); ++ i ) ret[ i ] = p_lhs OP p_rhs[ i ]; return ret; }

BINOPVM( + )
BINOPVM( - )
BINOPVM( * )
BINOPVM( / )

#define BINOPMV( OP ) \
template< typename T > Mat< T > operator OP( Mat< T > const & p_lhs, Vec< T > const & p_rhs ) { Mat< T > ret( p_lhs.size( ), Vec< T >( p_lhs[ 0 ] ) ); \
for( std::size_t r = 0; r < ret.size( ); ++ r ) for( std::size_t c = 0; c < p_lhs[ r ].size( ); ++ c ) ret[ r ][ c ] = p_lhs[ r ][ c ] OP p_rhs[ r ]; return ret; }

BINOPMV( + )
BINOPMV( - )
BINOPMV( * )
BINOPMV( / )

#define BINASSGNOPMV( OP ) \
template< typename T > Mat< T > & operator OP( Mat< T > & p_lhs, Vec< T > const & p_rhs ) { \
for( std::size_t r = 0; r < nrows( p_lhs ); ++ r ) for( std::size_t c = 0; c < ncols( p_lhs ); ++ c ) p_lhs[ r ][ c ] OP p_rhs[ r ]; return p_lhs; }

BINASSGNOPMV( += )
BINASSGNOPMV( -= )
BINASSGNOPMV( *= )
BINASSGNOPMV( /= )

template< typename T >
Mat< T >
t( Mat< T > const & p_mat ) {

	Mat< T >
	ret( ncols( p_mat ), Vec< T >( nrows( p_mat ) ) );

	for( std::size_t r = 0; r < nrows( ret ); ++ r ) {

		for( std::size_t c = 0; c < ncols( ret ); ++ c ) {

			ret[ r ][ c ] = p_mat[ c ][ r ];
		}
	}

	return ret;
}

template< typename T >
Mat< T >
operator ~( Mat< T > const & p_mat ) {

	return t( p_mat );
}

template< typename T >
Vec< T >
operator +( Vec< T > const & p_vec ) {

	return p_vec;
}

template< typename T >
Mat< T >
operator +( Mat< T > const & p_mat ) {

	return p_mat;
}

template< typename T >
Vec< T >
operator -( Vec< T > const & p_vec ) {

	Vec< T >
	ret( p_vec.size( ) );

	for( std::size_t  c = 0; c < ret.size( ); ++ c ) {

		ret[ c ] = -p_vec[ c ];
	}

	return ret;
}

template< typename T >
Mat< T >
operator -( Mat< T > const & p_mat ) {

	Mat< T >
	ret( nrows( p_mat ), Vec< T >( ncols( p_mat ) ) );

	for( std::size_t r = 0; r < nrows( ret ); ++ r ) {

		for( std::size_t c = 0; c < ncols( ret ); ++ c ) {

			ret[ r ][ c ] = - p_mat[ r ][ c ];
		}
	}

	return ret;
}


template< typename T >
Mat< T >
operator ^( Vec< T > const & p_lhs, Vec< T > const & p_rhs ) {

	Mat< T >
	ret;

	for( auto v : p_lhs )

		ret.push_back( v * p_rhs );

	return ret;
}

template< typename T >
Vec< T >
operator |( T const & p_lhs, Vec< T > const & p_rhs ) {

	return p_lhs * p_rhs;
}

template< typename T >
Vec< T >
operator |( Vec< T > const & p_lhs, T const & p_rhs ) {

	return p_lhs * p_rhs;
}

template< typename T >
Vec< T >
& operator |=( Vec< T > & p_lhs, T const & p_rhs ) {

	return p_lhs *= p_rhs;
}

template< typename T >
Mat< T >
operator |( T const & p_lhs, Mat< T > const & p_rhs ) {

	return p_lhs * p_rhs;
}

template< typename T >
Mat< T >
operator |( Mat< T > const & p_lhs, T const & p_rhs ) {

	return p_lhs * p_rhs;
}

template< typename T >
Mat< T >
& operator |=( Mat< T > & p_lhs, T const & p_rhs ) {

	return p_lhs = p_lhs | p_rhs;
}

template< typename T >
T
operator |( Vec< T > const & p_lhs, Vec< T > const & p_rhs ) {

	T
	s( 0 );

	for( std::size_t i = 0; i < p_lhs.size( ); ++ i  )

		s = s + p_lhs[ i ] * p_rhs[ i ];

	return s;
}

template< typename T >
Vec< T >
& operator |=( Vec< T > & p_lhs, Mat< T > const & p_rhs ) {

	p_lhs = p_lhs | p_rhs;

	return p_lhs;
}

template< typename T >
Vec< T >
operator |( Vec< T > const & p_lhs, Mat< T > const & p_rhs ) {

	Vec< T >
	ret( ncols( p_rhs ) );

	for( std::size_t c = 0; c < ncols( p_rhs ); ++ c ) {

		T
		sum = 0;

		for( std::size_t r = 0; r < nrows( p_rhs ); ++ r ) {

			sum += p_lhs[ r ] * p_rhs[ r ][ c ];
		}

		ret[ c ] = sum;
	}

	return ret;
}

template< typename T >
Vec< T >
operator |( Mat< T > const & p_lhs, Vec< T > const & p_rhs ) {

	Vec< T >
	ret( nrows( p_lhs ) );

	for( std::size_t r = 0; r < nrows( p_lhs ); ++ r ) {

		ret[ r ] = p_lhs[ r ] | p_rhs;
	}

	return ret;
}

template< typename T >
Mat< T >
operator |( Mat< T > const & p_lhs, Mat< T > const & p_rhs ) {

	Mat< T >
	ret( nrows( p_lhs ), Vec< T >( ncols( p_rhs ) ) );

	for( std::size_t r = 0; r < nrows( ret ); ++ r ) {

		for( std::size_t c = 0; c < ncols( ret ); ++ c ) {

			T
			s = 0;

			for( std::size_t i = 0; i < nrows( p_rhs ); ++ i ) {

				s += p_lhs[ r ][ i ] * p_rhs[ i ][ c ];
			}

			ret[ r ][ c ] =  s;
		}
	}

	return ret;
}

template< typename T >
Mat< T >
& operator |=( Mat< T > & p_lhs, Mat< T > const & p_rhs ) {

	p_lhs = p_lhs | p_rhs;

	return p_lhs;
}

template< typename T >
T
abs( Vec< T > const & p_vec ) {

	return sqrt( p_vec | p_vec );
}

template< typename T >
Mat< T >
eye( std::size_t const & p_rows ) {

	Mat< T >
	ret( p_rows, Vec< T >( p_rows ) );

	for( std::size_t r = 0; r < nrows( ret ); ++ r )

		for( std::size_t c = 0; c < ncols( ret ); ++ c )

			ret[ r ][ c ] = r == c ? T( 1 ) : T( 0 );

	return ret;
}

template< typename T >
Mat< T >
ones( std::size_t const & p_rows, std::size_t const & p_cols = 0 ) {

	return Mat< T >( p_rows, Vec< T >( p_cols == 0 ? p_rows : p_cols, 1 ) );
}

template< typename T >
Vec< T >
vrnd( std::size_t const & p_size ) {

	Vec< T >
	ret( p_size );

	for( auto & r : ret )

		r = static_cast< long double >( random( ) ) / RAND_MAX;

	return ret;
}

template< typename T >
Mat< T >
mrnd( std::size_t const & p_rows, std::size_t const & p_cols = 0 ) {

	Mat< T >
	ret( p_rows );

	for( auto & r : ret )

		r = vrnd< T >( p_cols < 1 ? p_rows : p_cols );

	return ret;
}

template< typename T >
std::size_t
maxWidthOfElements( Vec< T > const & p_vec ) {

	std::size_t
	len = 0,
	llen = 0;

	std::stringstream
	ss;

	for( auto v : p_vec ) {

		ss.str( "" );

		ss << v;

		llen = ss.str( ).length( );

		if( len < llen )

			len = llen;
	}

	return len;
}

template< typename T >
std::string
vec2Str( Vec< T > const & p_vec, int const & p_width = 0 ) {

	std::stringstream
	ss;

	for( auto v : p_vec ) {

		if( 0 < p_width )

			ss << std::setw( p_width );

		ss << v;
	}

	return ss.str( );
}

template< typename T >
std::ostream
& operator << ( std::ostream & p_os, Vec< T > const & p_vec ) {

	for( auto v : p_vec )

		p_os << v << "  ";

	return p_os;
}

template< typename T >
std::ostream
& operator << ( std::ostream & p_os, Mat< T > const & p_mat ) {

	std::size_t
	len = 0,
	llen = 0;

	for( auto v : p_mat ) {

		llen = maxWidthOfElements( v );

		if( len < llen )

			len = llen;
	}

	auto
	v = p_mat.cbegin( );

	while( v < p_mat.cend( ) - 1 )

		p_os << vec2Str( *v++, len + 1 ) << std::endl;

	p_os << vec2Str( *v, len + 1 );

	return p_os;
}

template< typename T >
std::ostream
& operator << ( std::ostream & p_os, Vec< Mat< T > > const & p_ten ) {

	for( auto v : p_ten )

		p_os << v << std::endl << std::endl;

	return p_os;
}

std::size_t
idxOfNxt( std::size_t const & p_val, std::size_t p_idx, Vec< std::size_t > const & p_vec ) {

	while( p_idx < p_vec.size( ) ) {

		if( p_vec[ p_idx ] == p_val ) {

			return p_idx;
		}

		++ p_idx;
	}

	return p_idx;
}

std::size_t
idxOf1st( std::size_t const & p_val, Vec< std::size_t > const & p_vec ) {

	return idxOfNxt( p_val, 0, p_vec );
}

Vec< std::size_t >
permutationZero( std::size_t const & p_size ) {

	Vec< std::size_t >
	permutation( p_size );

	for( std::size_t i = 0; i < p_size; ++ i ) {

		permutation[ i ] = i;
	}

	return permutation;
}

Vec< std::size_t >
& nextPermutation( Vec< std::size_t > & p_permutation ) {

	std::size_t
	curr = p_permutation.size( );

	while( 0 < curr -- ) {

		std::size_t
		greatest = idxOf1st( curr, p_permutation ),
		nextHole = idxOfNxt( p_permutation.size( ), greatest + 1, p_permutation );

		if( nextHole < p_permutation.size( ) ) {

			p_permutation[ nextHole ] = curr;
			p_permutation[ greatest ] = p_permutation.size( );

			nextHole = idxOfNxt( p_permutation.size( ), 0, p_permutation );

			while( nextHole < p_permutation.size( ) ) {

				p_permutation[ nextHole ] = ++ curr ;

				nextHole = idxOfNxt( p_permutation.size( ), 0, p_permutation );
			}

			return p_permutation;
		}

		p_permutation[ greatest ] = p_permutation.size( );
	}

	for( std::size_t i = 0; i < p_permutation.size( ); ++ i ) {

		p_permutation[ i ] = i;
	}

	return p_permutation;
}

std::size_t
fac( std::size_t p_n ) {

	return p_n == 0 ? 1 : p_n * fac( p_n - 1 );
}

template< typename T >
T
abs( Mat< T > const & p_m, Vec< std::size_t > & p_ridx, Vec< std::size_t > & p_cidx ) {

	if( nrows( p_m ) != ncols( p_m ) )

		return T( 0 );

	T
	d = T( 0 );

	std::size_t
	loops = fac( p_cidx.size( ) );

	char
	s01 = 0;

	Vec< std::size_t >
	idx = permutationZero( p_cidx.size( ) );

	for( std::size_t i = 0; i < loops; ++ i ) {

		T
		p = T( 1 );

		for( std::size_t j = 0; j < p_cidx.size( ); ++ j ) {

			p *= p_m[ p_ridx[ j ] ][ p_cidx[ idx[ j ] ] ];
		}

		d += ( ( ( ++ s01 ) & 0x2 ) == 0x2 ? -1 : 1 ) * p;

		idx = nextPermutation( idx );
	}

	return d;
}

template< typename T >
T
abs( Mat< T > const & p_m ) {

	Vec< std::size_t >
	cidx = permutationZero( ncols( p_m ) ),
	ridx = permutationZero( nrows( p_m ) );

	return abs( p_m, ridx, cidx );
}

template< typename T >
Mat< T >
inv( Mat< T > const & p_m ) {

	Mat< T >
	e = eye< T >( nrows( p_m ) ),
	m = p_m;

	for( std::size_t c = 0; c < ncols( m ) - 1; ++ c ) {

		std::size_t
		mr = c;

		for( std::size_t r = c + 1; r < ncols( m ); ++ r ) {

			if( abs( m[ mr ][ c ] ) < abs( m[ r ][ c ] ) ) {

				mr = r;
			}
		}

		Vec< T >
		tmp = m[ c ];

		m[ c ] = m[ mr ];
		m[ mr ] = tmp;

		tmp = e[ c ];
		e[ c ] = e[ mr ];
		e[ mr ] = tmp;
	}

	for( std::size_t c = 0; c < ncols( m ) - 1; ++ c ) {

		T
		a = 1. / m[ c ][ c ];

		for( std::size_t r = c + 1; r < nrows( m ); ++ r ) {

			T
			b = m[ r ][ c ];

			m[ r ] = m[ r ] - a * b * m[ c ];
			e[ r ] = e[ r ] - a * b * e[ c ];
		}
	}

	for( std::size_t c = ncols( m ) - 1; 0 < c; -- c ) {

		T
		a = 1. / m[ c ][ c ];

		for( int r = c - 1; 0 <= r; -- r ) {

			T
			b = m[ r ][ c ];

			m[ r ] = m[ r ] - a * b * m[ c ];
			e[ r ] = e[ r ] - a * b * e[ c ];
		}
	}

	for( std::size_t i = 0; i < nrows( e ); ++ i ) {

		e[ i ] = ( 1. / m[ i ][ i ] ) * e[ i ];
	}

	return e;
}

template< typename T >
Mat< T >
diag( Mat< T > const & p_m, bool p_s = false ) {

	Mat< T >
	e = eye< T >( nrows( p_m ) ),
	m = p_m;

	for( std::size_t c = 0; c < ncols( m ) - 1; ++ c ) {

		std::size_t
		mr = c;

		for( std::size_t r = c + 1; r < ncols( m ); ++ r ) {

			if( abs( m[ mr ][ c ] ) < abs( m[ r ][ c ] ) ) {

				mr = r;
			}
		}

		Vec< T >
		tmp = m[ c ];

		m[ c ] = m[ mr ];
		m[ mr ] = tmp;

		tmp = e[ c ];
		e[ c ] = e[ mr ];
		e[ mr ] = tmp;
	}

	for( std::size_t c = 0; c < ncols( m ) - 1; ++ c ) {

		T
		a = 1. / m[ c ][ c ];

		for( std::size_t r = c + 1; r < nrows( m ); ++ r ) {

			T
			b = m[ r ][ c ];

			m[ r ] = m[ r ] - a * b * m[ c ];
			e[ r ] = e[ r ] - a * b * e[ c ];
		}
	}

	for( std::size_t c = ncols( m ) - 1; 0 < c; -- c ) {

		T
		a = 1. / m[ c ][ c ];

		for( int r = c - 1; 0 <= r; -- r ) {

			T
			b = m[ r ][ c ];

			m[ r ] = m[ r ] - a * b * m[ c ];
			e[ r ] = e[ r ] - a * b * e[ c ];
		}
	}

	return p_s ? e : m;
}

double
round( double const & p_v, int const & p_digits = 0 ) {

	double
	r = exp10( -p_digits ) * std::round( exp10( p_digits ) * p_v );

	if( r * r < 1e-50 )

		r = +0;

	return r;
}

template< typename T >
T
round( T const & p_vec, int const & p_digits = 0 ) {

	T
	r = p_vec;

	for( auto & s : r ) {

		s = round( s, p_digits );
	}

	return r;
}

template < typename T >
Vec< T >
sub ( Vec< T > const & p_vec, int const & p_begin, int const & p_size ) {

	Vec< T >
	r;

	for( int i = p_begin; i < p_begin + p_size; ++ i ) {

		r.push_back( p_vec[ i ] );
	}

	return r;
}

template < typename T >
Mat< T >
sub ( Mat< T > const & p_mat, int const & p_row, int const & p_col, int const & p_rows, int const & p_cols ) {

	Mat< T >
	d;

	for( int i = p_row; i < p_row + p_rows; ++ i ) {

		d.push_back( sub( p_mat[ i ], p_col, p_cols ) );
	}

	return d;
}

template< typename T >
bool
save( std::string const & P_filename, Vec< T > const & p_vec ) {

	std::ofstream
	ofs( P_filename );

	if( ! ofs.is_open( ) )

		return false;

	ofs << p_vec;

	ofs.close( );

	return true;
}

template< typename T >
bool
load( std::ifstream & P_ifs, Vec< T > & p_vec ) {

	bool
	ret = false;

	p_vec.resize( 0 );

	std::string
	line;

	std::getline( P_ifs, line );

	std::stringstream
	ss( line );

	T
	val;

	while( ss >> val ) {

		p_vec.push_back( val );

		ret = true;
	}

	return ret;
}


template< typename T >
bool
load( std::ifstream & P_ifs, Mat< T > & p_vec ) {

	bool
	ret = false;

	p_vec.resize( 0 );

	Vec< T >
	v;

	while( load( P_ifs, v ) ) {

		p_vec.push_back( v );

		ret = true;
	}

	return ret;
}

template< typename T >
bool
load( std::ifstream & P_ifs, Vec< Vec< Vec< T > > > & p_ten ) {

	bool
	ret = false;

	p_ten.resize( 0 );

	Mat< T >
	v;

	while( load( P_ifs, v ) ) {

		p_ten.push_back( v );

		ret = true;
	}

	return ret;
}

template< typename T >
bool
load( std::string const & P_filename, Vec< T > & p_vec ) {

	std::ifstream
	ifs( P_filename );

	if( ! ifs.is_open( ) )

		return false;

	bool
	ret = load( ifs, p_vec );

	ifs.close( );

	return ret;
}

template< typename T >
bool
load( std::string const & P_filename, Mat< T > & p_mat ) {

	std::ifstream
	ifs( P_filename );

	if( ! ifs.is_open( ) )

		return false;

	bool
	ret = load( ifs, p_mat );

	ifs.close( );

	return ret;
}

template< typename T >
bool
load( std::string const & P_filename, Vec< Vec< Vec< T > > > & p_ten ) {

	std::ifstream
	ifs( P_filename );

	if( ! ifs.is_open( ) )

		return false;

	bool
	ret = load( ifs, p_ten );

	ifs.close( );

	return ret;
}

#endif // ALGEBRA_HPP
