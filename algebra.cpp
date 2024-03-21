#include "algebra.hpp"

namespace alg {

	bool
	gbit( char * p_bitset, SIZE const & p_bitId ) {

		SIZE
		byteId = p_bitId >> 3,
		bitId  = p_bitId & 0x7;

		return ( p_bitset[ byteId ] >> bitId ) & 1;
	}

	void
	cbit( char * p_bitset, SIZE const & p_bitId ) {

		SIZE
		byteId = p_bitId >> 3,
		bitId  = p_bitId & 0x7;

		p_bitset[ byteId ] &= static_cast<char>( ~( 1 << bitId ) );
	}

	void
	sbit( char * p_bitset, SIZE const & p_bitId ) {

		SIZE
		byteId = p_bitId >> 3,
		bitId  = p_bitId & 0x7;

		p_bitset[ byteId ] |= ( static_cast<char>(1 << bitId ) );
	}
}
