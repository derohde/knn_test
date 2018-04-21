/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <cmath>
#include <limits>

#include "knn_graph.hpp"
#include "random.hpp"
#include "oracle.hpp"

template <typename V = double>
class KNN_Tester {
    typedef typename KNN_Graph<V>::vertices_type vertices_type;
    
public:
    /**
     * tuning parameter c1 for psi, tuning parameter c2 for |T|
     */
    double c1 = 1.86, c2 = 1;

    KNN_Tester() {}
    
    /**
     * 
     * Calculates asymptotical upper bound for lattice sphere packing from Skoruppa07
     * @param Number of dimensions delta
     * @return estimate for c1
     * 
     */
    static double c1_approximate(const double delta) {
        auto epsilon = std::pow(10, -100);
        auto kissing_number_lattice_asymp_ub = std::exp(delta * (1 + epsilon)*(1 - std::log(2)) + delta/1.95);
        return std::log(kissing_number_lattice_asymp_ub) / std::log(2) / delta / 0.401 - 1;
    }
    
    /**
     * 
     * Property Testing Algorithm for k-nearest Neighborhood Graphs
     * @param KNN_Graph G, average degree of G, epsilon
     * @return true or false
     * 
     */
    virtual bool test(const KNN_Graph<V> &graph, const double d, const double epsilon = 0.001) const {
        const auto delta = graph.dimensions();
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
     * @param KNN_Graph G, average degree of G, epsilon
     * @return
     * 
     */
    bool test(const KNN_Graph<V> &graph, const double d, const double epsilon = 0.001) const {
        const auto delta = graph.dimensions();
        const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto psi = pow(2, 0.401 * delta * (1 + this->c1));
        const auto s = ceil(100 * k * sqrt(n) / epsilon * this->c2);
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
    /**
     * 
     * Property Testing Algorithm for k-Nearest Neighborhood Graphs - Graph Restauration
     * @param KNN_Graph G, average degree of G, epsilon
     * @return
     * 
     */
    auto improve(KNN_Graph<V> &graph, const double d, const double epsilon = 0.001) const {
        auto result = 0ul; 
        const auto delta = graph.dimensions();
        const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto psi = pow(2, 0.401 * delta * (1 + this->c1));
        const auto s = ceil(100 * k * sqrt(n) / epsilon * this->c2);
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
