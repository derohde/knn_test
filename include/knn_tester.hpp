/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <cmath>
#include <limits>

#include <boost/chrono/include.hpp>

#include "knn_graph.hpp"
#include "random.hpp"
#include "oracle.hpp"

class Tester_Result{
public:
    bool decision;
    double total_time;
    double query_time;
};

template <typename V = double>
class KNN_Tester {
    typedef typename KNN_Graph<V>::vertices_type vertices_type;
    
protected:
    bool auto_c1;
    
public:
    /**
     * 
     * tuning parameter c1 for psi
     * 
     */
    double c1 = 1;
    
    /**
     * 
     * tuning parameter c2 for |T|
     * 
     */
    double c2 = 1;

    KNN_Tester(const bool auto_c1 = true) : auto_c1{auto_c1} {}
    
    /**
     * 
     * Calculates approximate for c1
     * @param Number of dimensions delta
     * @return approximate for c1
     * 
     */
    static double c1_approximate(const KNN_Graph<V> &graph) {
        auto delta = graph.dimension();
        return std::pow(2, 0.401 * delta * (1 + 2.85 * delta / std::pow(delta, 1.4))); 
    }
    
    /**
     * 
     * Property Testing Algorithm for k-nearest Neighborhood Graphs
     * @param KNN_Graph G, average degree of G, epsilon
     * @return true or false
     * 
     */
    virtual Tester_Result test(const KNN_Graph<V> &graph, const double d, const double epsilon = 0.001) {
        if (auto_c1) this->c1 = c1_approximate(graph);
        const auto delta = graph.dimension();
        const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto s = std::ceil(100 * k * sqrt(n) / epsilon * c2);
        const auto t = std::ceil(log(10) * c1 * k * sqrt(n));
        
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
                auto dist = KNN_Graph<V>::euclidean_distance(v_value, graph.get_vertex(neighbor));
                if (dist > distN) {
                    distN = dist;
                }
            }
            for (unsigned long long j = 0; j < T.size(); ++j) {
                if (wrongly_connected_found) {
                    continue;
                }
                unsigned long long w = floor(T[j] * n);
                auto dist = KNN_Graph<V>::euclidean_distance(v_value, graph.get_vertex(w));
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
        Tester_Result result;
        if (wrongly_connected_found) {
            std::cout << "Reject!" << std::endl;
            std::cout << distw << " < " << distn << std::endl;
            result.decision = false;
        } else {
            std::cout << "Accept!" << std::endl;
            result.decision = true;
        }
        return result;
    }
    
    inline auto get_auto_c1() const {
        return auto_c1;
    }
    
    inline void set_auto_c1(const bool auto_c1) {
        this->auto_c1 = auto_c1;
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
    Tester_Result test(const KNN_Graph<V> &graph, const double d, const double epsilon = 0.001) {
        auto start = boost::chrono::process_real_cpu_clock::now();
        Oracle.reset_timer();
        if (this->auto_c1) this->c1 = this->c1_approximate(graph);
        const auto delta = graph.dimension();
        const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto s = std::ceil(100 * k * sqrt(n) / epsilon * this->c2);
        const auto t = std::ceil(log(10) * this->c1 * k * sqrt(n));
        
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
                auto dist = KNN_Graph<V>::euclidean_distance(v_value, neighbor);
                if (dist > distN) {
                    distN = dist;
                }
            }
            for (unsigned long long j = 0; j < T.size(); ++j) {
                if (wrongly_connected_found) {
                    continue;
                }
                unsigned long long w = floor(T[j] * n);
                auto dist = KNN_Graph<V>::euclidean_distance(v_value, graph.get_vertex(w));
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
        auto stop = boost::chrono::process_real_cpu_clock::now();
        auto total_time = (stop-start).count();
        auto query_time = Oracle.time();

        Tester_Result result;
        result.total_time = total_time / 1000000000.0;
        result.query_time = query_time / 1000000000.0;
        if (wrongly_connected_found) {
            std::cout << "Reject!" << std::endl;
            std::cout << distw << " < " << distn << std::endl;
            result.decision = false;
        } else {
            std::cout << "Accept!" << std::endl;
            result.decision = true;
        }
        return result;
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
    auto improve(KNN_Graph<V> &graph, const double d, const double epsilon = 0.001) {
        auto result = 0ul; 
        if (this->auto_c1) this->c1 = this->c1_approximate(graph);
        const auto delta = graph.dimension();
        const auto k = graph.get_k();
        const auto n = graph.number_vertices();
        const auto s = ceil(100 * k * sqrt(n) / epsilon * this->c2);
        const auto t = ceil(log(10) * this->c1 * k * sqrt(n));
        
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
                auto dist = KNN_Graph<V>::euclidean_distance(v_value, graph.get_vertex(neighbor));
                if (dist > distN) {
                    distN = dist;
                    furthest = neighbor;
                }
            }
            for (unsigned long long j = 0; j < T.size(); ++j) {
                unsigned long long w = floor(T[j] * n);
                auto dist = KNN_Graph<V>::euclidean_distance(v_value, graph.get_vertex(w));
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
                                auto distF = KNN_Graph<V>::euclidean_distance(v_value, graph.get_vertex(neighbor));
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
