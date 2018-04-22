/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <cxxabi.h>
#include <iostream>

#include <boost/python/numpy.hpp>

#include "d_dim_tuple.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

template <typename T = double>
class Relation : public std::vector<Tuple<T>> {
    typedef std::vector<Tuple<T>> relation_type;
    typedef unsigned long long index_type;
    
public:
    Relation() : relation_type{} {}
    Relation(const index_type n) : relation_type(n) {}
    Relation(const np::ndarray &in) : relation_type{} {
        auto dimensions = in.get_nd();
        if (dimensions != 2 or in.get_dtype() != np::dtype::get_builtin<T>()) {
            int status;
            const std::type_info &ti = typeid(T);
            std::cerr << "Need 2-dimensional numpy array of type " << abi::__cxa_demangle(ti.name(), 0, 0, &status) << " !" << std::endl;
            return;
        }
        auto size = in.get_shape();
        auto strides = in.get_strides();
        auto data = in.get_data();
        for (index_type i = 0; i < size[0]; ++i) {
            Tuple<T> newtuple{static_cast<unsigned long>(size[1])};
            for (unsigned long j = 0; j < size[1]; ++j) {
                newtuple[j] = *reinterpret_cast<const T*>(data + i * strides[0] + j * strides[1]);
            }
            this->push_back(newtuple);
        }
    }
    
    inline auto dimension() const {
        return this->size() == 0 ? 0 : this->operator[](0).dimension();
    }
    
    inline auto as_ndarray() const {
        p::list l;
        np::dtype dt = np::dtype::get_builtin<T>();
        np::ndarray result = np::array(l, dt);
        index_type size = this->size();
        if (size == 0) {
            return result;
        }
        unsigned long d = this->get(0).size();
        for (index_type i = 0; i < size; ++i) {
            p::list lu;
            for (unsigned long j = 0; j < d; ++j) {
                lu.append((*this)[i][j]);
            }
            l.append(lu);
        }
        result = np::array(l, dt);
        return result;
    }
    
    inline auto get(const index_type i) const {
        if (i >= this->size()) {
            throw std::invalid_argument("i >= relation size");
        }
        return this->operator[](i);
    }
    
    inline auto as_str() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
    
    inline auto repr() const {
        std::stringstream ss;
        ss << "Relation of Dimension " << dimension() << ":" << std::endl;
        ss << as_str();
        return ss.str();
    }
    
    inline auto pbegin() const {
        return this->begin();
    }
    
    inline auto pend() const {
        return this->end();
    }
};

template <typename T>
std::ostream& operator<<(std::ostream &out, const Relation<T> &r) {
    const auto size = r.size();
    out << "{";
    for (auto i = size-size; i < size-1; ++i) {
        out << r.get(i);
        out << std::endl;
    }
    out << r.get(size-1) << "}";
    return out;
}
