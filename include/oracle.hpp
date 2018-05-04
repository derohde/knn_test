/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <iostream>

#include <boost/chrono/include.hpp>
#include <boost/python/numpy.hpp>

#include "relation.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

template <typename T = double>
class Query_Oracle {
    typedef unsigned long long index_type;
    
    p::object function;
    unsigned long long query_time;

public:
    explicit Query_Oracle(PyObject *callable) : query_time(0) {
        try {
            index_type v = 0;
            auto result = p::call<np::ndarray>(callable, v);
            if (result.get_nd() != 2) {
                throw std::invalid_argument("Function must return 2-dim. ndarray of correct type!");
            }
            if (result.get_dtype() != np::dtype::get_builtin<T>()) {
                throw std::invalid_argument("Function must return 2-dim. ndarray of correct type!");
            }
            function = p::object{p::handle<>(p::borrowed(callable))};
        } catch (std::exception &e) {
            std::cerr << "Need a function f: {0, ..., n-1} -> P(R^d)!" << std::endl;
            std::cerr << e.what() << std::endl;
        }
    }
    
    auto query(const index_type i) {
        p::list l;
        np::ndarray result{np::array(l, np::dtype::get_builtin<T>())};
        auto start = boost::chrono::thread_clock::now();
        try {
            result = p::call<np::ndarray>(function.ptr(), i);
        } catch (std::exception &e) {
            std::cerr << e.what() << std::endl;
        }
        auto stop = boost::chrono::thread_clock::now();
        auto duration = (stop-start).count();
        query_time += static_cast<unsigned long long>(duration);
        return Relation<T>(result);
    }
    
    void reset_timer() {
        query_time = std::chrono::nanoseconds(0);
    }

    unsigned long long time() const {
        return query_time;
    }
};
