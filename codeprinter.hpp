#ifndef CODEPRINTER_HPP
#define CODEPRINTER_HPP

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

class CodePrinter {

	private:

			std::size_t
			tab_size;

	public:

		CodePrinter(std::string const &p_filename, int const  & p_tab_size = 4) :
		tab_size(p_tab_size) {

			std::ifstream
			fs(p_filename);

			if(fs.is_open()) {

				std::stringstream
				ss;

				ss << fs.rdbuf();

				fs.close();

				std::string
				s = ss.str();

				std::size_t
				from = 0,
				to = s.find("//@", from);

				while(to != std::string::npos) {

					from = to + 3;

					std::size_t
					send = s.find("\n", to);

					std::string
					name = std::string(s, from, send - from);

					to = s.find("//@", send + 1);

					if(0 < name.length()) {

						text[name] = std::string(s, send + 1, to - send - 1);
					}
				}
			}
		}

	public:

		std::map< std::string, std::string >
		text;
		//from https://stackoverflow.com/questions/2896600/how-to-replace-all-occurrences-of-a-character-in-string
		std::string replace_tabs(std::string str, const std::string & from = "\t", const std::string& to = "    ") {

			std::size_t start_pos = 0;

			while((start_pos = str.find(from, start_pos)) != std::string::npos) {

				str.replace(start_pos, from.length(), to);

				start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
			}

			return str;
		}

		void
		print(std::string const & p_snippetname, int const & p_tab_size = -1) {

			if(0 <= p_tab_size) {

				tab_size = p_tab_size;
			}

			std::string
			txt = text[p_snippetname];

			txt = replace_tabs(txt, "\t", std::string(tab_size, ' '));

			std::cout <<
				"----------------------------------------------------------------------------------------------------------------" <<
				"\n[" <<
				p_snippetname <<
				"]\nc++ - code:\n" <<
				txt <<
				"\nout:\n";
		}

		static bool
		waitForENTER() {

			std::cout << "Press [ENTER] for next show!\n";

			int
			c = std::cin.get();

			return (c == 'x') || (c == 'X') || (c == 'q') || (c == 'Q');
		}

		static bool
		WFE () {

			if(waitForENTER())
				exit(-1);

			return true;
		}
};

#endif // CODEPRINTER_HPP
