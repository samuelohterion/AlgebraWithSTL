#ifndef CODEPRINTER_HPP
#define CODEPRINTER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

class CodePrinter {

	public:

		CodePrinter( std::string const &p_filename ) {

			std::ifstream
			fs( p_filename );

			if( fs.is_open( ) ) {

				std::stringstream
				ss;

				ss << fs.rdbuf( );

				fs.close( );

				std::string
				s = ss.str( );

				std::size_t
				from = 0,
				to = s.find( "//@", from );

				while( to != std::string::npos ) {

					from = to + 3;

					std::size_t
					send = s.find( "\n", to );

					std::string
					name = std::string( s, from, send - from );

					to = s.find( "//@", send + 1 );

					if( 0 < name.length( ) ) {

						text[ name ] = std::string( s, send + 1, to - send - 1 );
					}
				}
			}
		}

	public:

		std::map< std::string, std::string >
		text;

		void
		print( std::string const & p_snippetname ) {

			std::cout <<
				"----------------------------------------------------------------------------------------------------------------" <<
				"\n[" <<
				p_snippetname <<
				"]\nc++:\n" <<
				text[ p_snippetname ] <<
				"\nout:\n";
		}
};

#endif // CODEPRINTER_HPP
