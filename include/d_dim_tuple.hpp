/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <iostream>
#include <limits>
#include <sstream>
#include <cmath>

#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace p = boost::python;

template <typename T = double>
class Tuple : public std::vector<T> {   
    typedef unsigned long index_type;
     
public:
    typedef T value_type;
    typedef std::vector<value_type> tuple_type;
    
    Tuple() : tuple_type{} {}
    Tuple(const index_type d) : tuple_type(d) {}
    Tuple(const T &in, const index_type d) : tuple_type(d, in) {}
    Tuple(const tuple_type &in, const index_type d) : tuple_type{} {
        if (in.size() < d) {
            throw std::invalid_argument("Could not build d-tuple from vector of dimension "+std::to_string(in.size()));
        }
        this->assign(in.begin(), in.begin() + d);
    }
    
    Tuple(const np::ndarray &in) : tuple_type{} {
        auto dimensions = in.get_nd();
        if (dimensions != 1 or in.get_dtype() != np::dtype::get_builtin<T>()) {
            int status;
            const std::type_info &ti = typeid(T);
            std::cerr << "Need 1-dimensional numpy array of type " << abi::__cxa_demangle(ti.name(), 0, 0, &status) << " !" << std::endl;
            return;
        }
        auto size = in.get_shape();
        auto strides = in.get_strides();
        auto data = in.get_data();
        for (index_type i = 0; i < size[0]; ++i) {
            (*this)[i] = *reinterpret_cast<const T*>(data + i);
        }
    }
    
    inline auto get(const index_type i) const {
        if (i >= this->size()) return std::numeric_limits<T>::signaling_NaN();
        return this->operator[](i);
    }
    
    inline auto dimension() const {
        return this->size();
    }
    
    inline auto operator-(const Tuple<T> &o) const {
        Tuple<T> result{*this, this->size()};
        if (this->size() != o.size()) {
            result = Tuple<T>(std::numeric_limits<T>::signaling_NaN(), this->size());
            return result;
        }
        for (index_type i = 0; i < this->size(); ++i) {
            result[i] -= o.get(i);
        }
        return result;
    }
    
    inline auto operator*(const Tuple<T> &o) const {
        if (this->size() != o.size()) return std::numeric_limits<T>::signaling_NaN();
        T sum = 0;
        for (index_type i = 0; i < this->size(); ++i) {
            sum += get(i) * o.get(i);
        }
        return sum;
    }
    
    inline auto operator==(const Tuple<T> &o) const {
        if (this->size() != o.size()) {
            std::cerr << "Can not compare tuples of different dimension!" << std::endl;
            return false;
        }
        for (index_type i = 0; i < this->size(); ++i) {
            if (std::fabs(get(i) - o.get(i)) < std::numeric_limits<T>::epsilon()) continue;
            else return false;
        }
        return true;
    }
    
    inline bool operator<(const Tuple<T> &o) const {
        if (this->size() != o.size()) {
            std::cerr << "Can not compare tuples of different dimension!" << std::endl;
            return false;
        }
        for (index_type i = 0; i < this->size(); ++i) {
            if (not (std::fabs(get(i) - o.get(i)) < std::numeric_limits<T>::epsilon())) {
                if (o.get(i) - get(i) >= std::numeric_limits<T>::epsilon()) {
                    return true;
                } else {
                    return false;
                }
            }
        }
        return false;
    }
    
    inline auto as_str() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
    
    inline auto repr() const {
        std::stringstream ss;
        ss << "Tuple of Dimension " << dimension() << ":" << std::endl;
        ss << as_str();
        return ss.str();
    }
    
    inline auto as_ndarray() const {
        np::dtype dt = np::dtype::get_builtin<T>();
        p::list l;
        np::ndarray result = np::array(l, dt);
        for (const auto &elem: *this) {
            l.append(elem);
        }
        result = np::array(l, dt);
        return result;
    }
    
    inline auto pbegin() const {
        return this->begin();
    }
    
    inline auto pend() const {
        return this->end();
    }
};

template <typename T>
std::ostream& operator<<(std::ostream &out, const Tuple<T> &t) {
    const auto d = t.size();
    out << "(";
    for (auto i = d-d; i < d-1; ++i) {
        out << t.get(i);
        out << ", ";
    }
    out << t.get(d-1) << ")";
    return out;
}
