/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <random>

#include "relation.hpp"

template <typename T = double>
class Uniform_Random_Generator {
    std::mt19937_64 mersenne_twister;
    std::uniform_real_distribution<T> distribution;
    T lbound, ubound;
    
public:
    Uniform_Random_Generator(const T lbound = 0, const T ubound = 1) : lbound{lbound}, ubound{ubound}, mersenne_twister{std::random_device{}()}, distribution{lbound, ubound} {}
    
    inline T get() {
        return distribution(mersenne_twister);
    }
    
    inline std::vector<T> get(const unsigned long n) {
        std::vector<T> results;
        for (auto i = n; i > 0; --i) {
            results.push_back(get());
        }
        return results;
    }
};

template <typename T = double>
class Uniform_Random_Tuple_Generator : public Uniform_Random_Generator<T> {
    unsigned long long n;
    unsigned long d;
    
public:
    Uniform_Random_Tuple_Generator() {}
    Uniform_Random_Tuple_Generator(const unsigned long long n, const unsigned long d = 3) : d{d}, n{n} {}
    
    Relation<T> get() {
        Relation<T> result(n);
        #ifndef SINGLETHREAD
        #pragma omp parallel for shared(result)
        #endif
        for (unsigned long long i = 0; i < n; ++i) {
            Tuple<T> newtuple{d};
            for (unsigned j = 0; j < d; ++j) {
                newtuple[j] = this->Uniform_Random_Generator<T>::get();
            }
            result[i] = newtuple;
        }
        return result;
    }
};
