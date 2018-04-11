/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <boost/tokenizer.hpp>

#include "relation.hpp"

template <typename T = double>
Relation<T> read_csv(std::string filename, const unsigned long d) {
    std::ifstream in(filename.c_str());
    Relation<T> result;
    std::vector<double> vec;
    std::string line;
    unsigned long i = 1;

    while (std::getline(in, line)) {
        if (line[0] == '#') continue;
        vec.clear();
        boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
        for (const auto &token: tok) {
            try {
                vec.push_back(std::stof(token));
            } catch (std::invalid_argument &e) {
                std::cerr << e.what() << std::endl;
            }
        }
        try {
            result.push_back(Tuple<T>(vec, d));
        } catch (std::invalid_argument) {
            std::cerr << "line " << i << " does not contain enough data!" << std::endl;
        }
        ++i;
    }
    std::cout << "Imported " << result.size() << " " << d << "-tuples." << std::endl;
    return result;
}
