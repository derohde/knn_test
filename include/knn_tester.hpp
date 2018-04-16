/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <cmath>
#include <limits>

#include "knn_graph.hpp"
#include "random.hpp"
#include "oracle.hpp"

template <typename V = double>
class KNN_Tester {
    typedef typename KNN_Graph<V>::vertices_type vertices_type;
    
public:
    KNN_Tester() {}
    
    double calculate_c1_lower_bound(const double delta) const {
        #if __GNUC__ >= 7
        const long double pi = std::acos(-1);
        const long double kissing_number_lattice_lb_density = std::riemann_zeta(delta) / std::pow(2, delta-1);
        const long double sphere_volume = 2 * std::pow(pi, delta / 2) / std::tgamma(delta / 2) / delta;
        const long double hermite = 4 * std::pow(kissing_number_lattice_lb_density / sphere_volume, 2 / delta);
        const long double kissing_number_lattice_lb = std::pow(hermite, delta);
        return std::log(kissing_number_lattice_lb) / std::log(2) / delta / 0.401 - 1;
        #else
        std::cerr << "This function was not implemented, because gcc version was < 7!" << std::endl;
        return 1;
        #endif
    }
    
    /**
     * 
     * Property Testing Algorithm for k-nearest Neighborhood Graphs
     * @param KNN_Graph G, average degree of G, dimension of G's vertices, epsilon, tuning parameter c1 for psi, tuning parameter c2 for |T|
     * @return true or false
     * 
     */
    virtual bool test(const KNN_Graph<V> &graph, const double d, const unsigned long delta = 3, const double epsilon = 0.001, const double c1 = 1.86, const double c2 = 1.0) const {
        const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto psi = pow(2, 0.401 * delta * (1 + c1));
        const auto s = ceil(100 * k * sqrt(n) / epsilon * c2);
        const auto t = ceil(log(10) * psi * k * sqrt(n));
        
        Uniform_Random_Generator<double> urandom_gen;
        
        const auto S = urandom_gen.get(s);
        const auto T = urandom_gen.get(t);
        
        std::cout << "|S| = " << s << std::endl;
        std::cout << "|T| = " << t << std::endl;
        
        bool wrongly_connected_found = false;
        double distn, distw;
        
        #pragma omp parallel for shared(wrongly_connected_found, distn, distw)
        for (unsigned long long i = 0; i < S.size(); ++i) {
            if (wrongly_connected_found) continue;
            const unsigned long long v = floor(S[i] * n);
            const auto v_value = graph.get_vertex(v);
            if (graph.number_neighbors(v) > 100 * k * d / epsilon) continue;
            const auto &neighbors = graph.get_edges()[v];
            V distN = 0;
            for (const auto &neighbor: neighbors) {
                auto dist = KNN_Graph<V>::euclidean_distance_squared(v_value, graph.get_vertex(neighbor));
                if (dist > distN) {
                    distN = dist;
                }
            }
            for (unsigned long long j = 0; j < T.size(); ++j) {
                unsigned long long w = floor(T[j] * n);
                if (wrongly_connected_found) {
                    continue;
                }
                auto dist = KNN_Graph<V>::euclidean_distance_squared(v_value, graph.get_vertex(w));
                if (v != w and distN - dist > std::numeric_limits<V>::epsilon()) {
                    auto is_neighbor = false;
                    for (const auto &neighbor: neighbors) {
                        if (w == neighbor) {
                            is_neighbor = true;
                            break;
                        }
                    }
                    if (not is_neighbor) {
                        #pragma omp critical
                        {
                            if (not wrongly_connected_found) {
                                wrongly_connected_found = true;
                                distn = distN;
                                distw = dist;
                            }
                        }
                    }
                }
            }
        }
        if (wrongly_connected_found) {
            std::cout << "Reject!" << std::endl;
            std::cout << distw << " < " << distn << std::endl;
            return false;
        } else {
            std::cout << "Accept!" << std::endl;
            return true;
        }
    }
};

template <typename V = double>
class KNN_Tester_Oracle : public KNN_Tester<V> {
    Query_Oracle<V> Oracle;
    
public:
    KNN_Tester_Oracle(const Query_Oracle<V> &oracle) : Oracle{std::move(oracle)} {}

    /**
     * 
     * Property Testing Algorithm for k-Nearest Neighborhood Graphs - Oracle Version
     * @param KNN_Graph G, average degree of G, dimension of G's vertices, epsilon, tuning parameter c1 for psi, tuning parameter c2 for |T|
     * @return
     * 
     */
    bool test(const KNN_Graph<V> &graph, const double d, const unsigned long delta = 3, const double epsilon = 0.001, const double c1 = 1.86, const double c2 = 1) const {
    const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto psi = pow(2, 0.401 * delta * (1 + c1));
        const auto s = ceil(100 * k * sqrt(n) / epsilon * c2);
        const auto t = ceil(log(10) * psi * k * sqrt(n));
        
        Uniform_Random_Generator<double> urandom_gen;
        
        const auto S = urandom_gen.get(s);
        const auto T = urandom_gen.get(t);
        
        std::cout << "|S| = " << s << std::endl;
        std::cout << "|T| = " << t << std::endl;
        
        bool wrongly_connected_found = false;
        double distn, distw;
        
        #pragma omp parallel for shared(wrongly_connected_found, distn, distw)
        for (unsigned long long i = 0; i < S.size(); ++i) {
            if (wrongly_connected_found) continue;
            const unsigned long long v = floor(S[i] * n);
            const auto v_value = graph.get_vertex(v);
            Relation<V> neighbors;
            #pragma omp critical
            {
                neighbors = Oracle.query(v);
            }
            if (neighbors.size() > 100 * k * d / epsilon) continue;
            V distN = 0;
            for (const auto &neighbor: neighbors) {
                auto dist = KNN_Graph<V>::euclidean_distance_squared(v_value, neighbor);
                if (dist > distN) {
                    distN = dist;
                }
            }
            for (unsigned long long j = 0; j < T.size(); ++j) {
                unsigned long long w = floor(T[j] * n);
                if (wrongly_connected_found) {
                    continue;
                }
                auto dist = KNN_Graph<V>::euclidean_distance_squared(v_value, graph.get_vertex(w));
                if (v != w and distN - dist > std::numeric_limits<V>::epsilon()) {
                    auto is_neighbor = false;
                    for (const auto &neighbor: neighbors) {
                        if (graph.get_vertex(w) == neighbor) {
                            is_neighbor = true;
                            break;
                        }
                    }
                    if (not is_neighbor) {
                        #pragma omp critical
                        {
                            if (not wrongly_connected_found) {
                                wrongly_connected_found = true;
                                distn = distN;
                                distw = dist;
                            }
                        }
                    }
                }
            }
        }
        if (wrongly_connected_found) {
            std::cout << "Reject!" << std::endl;
            std::cout << distw << " < " << distn << std::endl;
            return false;
        } else {
            std::cout << "Accept!" << std::endl;
            return true;
        }
    }
};

template <typename V = double>
class KNN_Improver : public KNN_Tester<V> {
public:
    auto improve(KNN_Graph<V> &graph, const double d, const unsigned long delta = 3, const double epsilon = 0.001, const double c1 = 1.86, const double c2 = 1.0) const {
        auto result = 0ul; 
        const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto psi = pow(2, 0.401 * delta * (1 + c1));
        const auto s = ceil(100 * k * sqrt(n) / epsilon * c2);
        const auto t = ceil(log(10) * psi * k * sqrt(n));
        
        Uniform_Random_Generator<double> urandom_gen;
        
        const auto S = urandom_gen.get(s);
        const auto T = urandom_gen.get(t);
        
        std::cout << "|S| = " << s << std::endl;
        std::cout << "|T| = " << t << std::endl;
        
        #pragma omp parallel for shared(result)
        for (unsigned long long i = 0; i < S.size(); ++i) {
            const unsigned long long v = floor(S[i] * n);
            const auto v_value = graph.get_vertex(v);
            if (graph.number_neighbors(v) > 100 * k * d / epsilon) continue;
            auto &neighbors = graph.get_edges()[v];
            V distN = 0;
            unsigned long long furthest = 0;
            for (const auto &neighbor: neighbors) {
                auto dist = KNN_Graph<V>::euclidean_distance_squared(v_value, graph.get_vertex(neighbor));
                if (dist > distN) {
                    distN = dist;
                    furthest = neighbor;
                }
            }
            for (unsigned long long j = 0; j < T.size(); ++j) {
                unsigned long long w = floor(T[j] * n);
                auto dist = KNN_Graph<V>::euclidean_distance_squared(v_value, graph.get_vertex(w));
                if (v != w and distN - dist > std::numeric_limits<V>::epsilon()) {
                    auto is_neighbor = false;
                    for (const auto &neighbor: neighbors) {
                        if (w == neighbor) {
                            is_neighbor = true;
                            break;
                        }
                    }
                    if (not is_neighbor) {
                        #pragma omp critical
                        {
                            graph.get_edges()[v][furthest] = w;
                            ++result;
                            distN = 0;
                            for (const auto &neighbor: neighbors) {
                                auto distF = KNN_Graph<V>::euclidean_distance_squared(v_value, graph.get_vertex(neighbor));
                                if (distF > distN) {
                                    distN = distF;
                                    furthest = neighbor;
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
};
