/*
Copyright 2018 Dennis Rohde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>

#include <boost/python/numpy.hpp>

#include "relation.hpp"

namespace np = boost::python::numpy;

template <typename T = double>
class KNN_Graph {    
public:
    typedef unsigned long long index_type;

    class Adjacency_List : public std::vector<index_type> {
    public:
        inline auto get(const index_type i) const {
            if (i >= this->size()) return std::numeric_limits<index_type>::signaling_NaN();
            return (*this)[i];
        }
        
        inline auto length() const {
            return this->size();
        }
        
        inline auto pbegin() const {
            return this->begin();
        }
        
        inline auto pend() const {
            return this->end();
        }
    };
    
    typedef Tuple<T> location_type;
    typedef Relation<T> vertices_type;
    typedef Adjacency_List adjacency_list_type;
    typedef std::vector<adjacency_list_type> edges_type;
    
    inline static T euclidean_distance(const location_type &a, const location_type &b) {
        auto diff = a - b;
        return sqrt(diff * diff);
    }
    
    inline static T euclidean_distance_squared(const location_type &a, const location_type &b) {
        auto diff = a - b;
        return diff * diff;
    }
    
    KNN_Graph() : k{0}, vertices{}, edges{} {}
    
    KNN_Graph(const unsigned long k) : k{k}, vertices{}, edges{} {}
    
    void sort() {
        std::vector<index_type> index(number_vertices());
        
        #pragma omp parallel for shared(index)
        for (index_type i = 0; i < number_vertices(); ++i) {
            index[i] = i;
        }
        
        auto less = [&](const index_type lhs, const index_type rhs) -> bool {
            return vertices[lhs] < vertices[rhs];
        };
        
        std::sort(index.begin(), index.end(), less);
        
        vertices_type new_vertices(number_vertices());
        edges_type new_edges(number_vertices());
        
        #pragma omp parallel for shared(new_vertices, new_edges, index)
        for (index_type i = 0; i < number_vertices(); ++i) {
            new_vertices[i] = vertices[index[i]];
            new_edges[i] = edges[index[i]];
            std::sort(new_edges[i].begin(), new_edges[i].end(), less);
        }
        
        vertices = new_vertices;
        edges = new_edges;
    }
    
    auto epsilon(const KNN_Graph<T> &hp) const {
        double epsilon = 0.0;
        auto g = *this;
        auto h = hp;
        
        g.sort();
        h.sort();
        
        index_type i = 0, j = 0;
        unsigned long long denominator = 0;
        
        while (i < g.number_vertices() and j < h.number_vertices()) {
            if (g.get_vertex(i) == h.get_vertex(j)) {
                auto g_neighbors = g.get_neighbors(i);
                auto h_neighbors = h.get_neighbors(j);
                auto g_number_neighbors = g_neighbors.size();
                auto h_number_neighbors = h_neighbors.size();
                denominator += g_number_neighbors;
                index_type k = 0, l = 0;
                while (k < g_number_neighbors and l < h_number_neighbors) {
                    if (g_neighbors[k] < h_neighbors[l]) {
                        epsilon += 1;
                        ++k;
                    } else if (g_neighbors[k] > h_neighbors[l]) {
                        ++l;
                    } else {
                        ++k;
                        ++l;
                    }
                }
                epsilon += g_number_neighbors - k;
                ++i;
                ++j;
            } else if (g.get_vertex(i) < h.get_vertex(j)) {
                auto number_neighbors = g.get_neighbors(i).size();
                epsilon += number_neighbors;
                denominator += number_neighbors;
                ++i;
            } else {
                ++j;
            }
        }
        
        for(; i < g.number_vertices(); ++i) {
            auto number_neighbors = g.get_neighbors(i).size();
            epsilon += number_neighbors;
            denominator += number_neighbors;
        }

        epsilon /= denominator;
        
        return epsilon;
    }
    
    inline auto dimension() const {
        return number_vertices() == 0 ? 0 : vertices[0].dimension();
    }
    
    inline auto number_vertices() const {
        return vertices.size();
    }
    
    inline auto number_edges() const {
        return edges_number;
    }
    
    inline auto number_wrongly_connected_vertices() const {
        unsigned long long result = 0;
        
        #pragma omp parallel for shared(result)
        for (index_type i = 0; i < number_vertices(); ++i) {
            T distN = 0;
            auto wrongly_connected = false;
            auto &adj_list = edges[i];
                        
            for (index_type j = 0; j < adj_list.size(); ++j) {
                auto dist = euclidean_distance_squared(vertices[i], vertices[adj_list[j]]);
                if (std::fabs(dist - distN) > std::numeric_limits<T>::epsilon() and dist > distN) {
                    distN = dist;
                }
            }
            
            for (index_type j = 0; j < number_vertices(); ++j) {
                if (wrongly_connected) continue;
                else if (std::find(adj_list.begin(), adj_list.end(), j) == adj_list.end() and i != j) {
                    auto dist = euclidean_distance_squared(vertices[i], vertices[j]);
                    if (std::fabs(dist - distN) > std::numeric_limits<T>::epsilon() and dist < distN) {
                        #pragma omp critical
                        {
                            if (not wrongly_connected) {
                                ++result;
                                wrongly_connected = true;
                            }
                        }
                    }
                }
            }
        } 
        return result;
    }
    
    inline auto get_vertex(const index_type i) const {
        return vertices[i];
    }
    
    inline auto get_neighbors(const index_type i) const {
        return edges[i];
    }
    
    inline auto number_neighbors(const index_type i) const {
        return edges[i].size();
    }
    
    inline auto get_k() const {
        return k;
    }
    
    virtual void build(const vertices_type &vertices) {
        this->vertices = vertices;
        this->edges = edges_type(vertices.size());
    }
    
    virtual void build(const np::ndarray &in) {
        this->vertices = vertices_type{in};
    }
    
    void edges_from_ndarray(const np::ndarray &in) {
        auto dimensions = in.get_nd();
        if (dimensions != 2 or in.get_dtype() != np::dtype::get_builtin<bool>()) {
            std::cerr << "Need 2-dimensional numpy array of type bool!" << std::endl;
            return;
        }
        this->edges = edges_type(vertices.size());
        auto size = in.get_shape();
        auto strides = in.get_strides();
        auto data = in.get_data();
        bool adjacent = false;
        for (index_type i = 0; i < size[0]; ++i) {
            for (index_type j = 0; j < size[1]; ++j) {
                adjacent = *reinterpret_cast<const bool*>(data + i * strides[0] + j * strides[1]);
                if (adjacent) {
                    this->edges[i].push_back(j);
                    ++this->edges_number;
                }
            }
        }
    }
    
    inline const auto& get_edges() const {
        return edges;
    }
    
    inline const auto& get_vertices() const {
        return vertices;
    }
    
    inline auto& get_edges() {
        return edges;
    }
    
    inline auto& get_vertices() {
        return vertices;
    }
    
    inline auto edges_begin() const {
        return edges.begin();
    }
    
    inline auto edges_end() const {
        return edges.end();
    }
    
    inline auto vertices_begin() const {
        return vertices.begin();
    }
    
    inline auto vertices_end() const {
        return vertices.end();
    }
    
    inline auto as_str() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
    
    inline auto repr() const {
        std::stringstream ss;
        ss << get_k() <<"-Nearest Neighbor Graph of Dimension " << dimension() << ":" << std::endl;
        ss << as_str();
        return ss.str();
    }
    
protected:
    unsigned long k;
    unsigned long long edges_number = 0;
    vertices_type vertices;
    edges_type edges;

};

template <typename T = double>
class KNN_Graph_Exact : public KNN_Graph<T> {
    typedef KNN_Graph<T> super;
    typedef typename super::index_type index_type;

public:
    KNN_Graph_Exact(const unsigned long k = 10) {
        this->k = k;
        this->vertices = typename super::vertices_type{};
        this->edges = typename super::edges_type{};
    }
    
    void build(const np::ndarray &in) {
        this->build(typename super::vertices_type(in));
    }
    
    void build(const typename super::vertices_type &vertices) {
        this->vertices = vertices;
        this->edges = typename super::edges_type(vertices.size());
        std::cout << "Building exact " << this->k << "-NNGraph with " << vertices.size() << " vertices:" << std::endl;

        auto n = vertices.size();
        
        for (index_type i = 0; i < n; ++i) {
            auto distance_furthest = std::numeric_limits<double>::infinity();
            
            for (index_type j = 0; j < n; ++j) {
                
                if (i != j) {
                
                    auto dist = this->euclidean_distance_squared(this->vertices[i], this->vertices[j]);
                                        
                    if (this->edges[i].length() >= this->k) {
                        if (dist < distance_furthest) {
                            index_type furthest = 0;
                            index_type neighbor = 0;
                            index_type furthest_neighbor = 0;
                            
                            for (index_type l = 0; l < this->edges[i].length(); ++l) {
                                neighbor = this->edges[i][l];
                                furthest_neighbor = this->edges[i][furthest];
                                
                                if (this->euclidean_distance_squared(this->vertices[i], this->vertices[furthest_neighbor]) < this->euclidean_distance_squared(this->vertices[i], this->vertices[neighbor])) {
                                    furthest = l;
                                }
                            }
                            this->edges[i][furthest] = j;
                            
                            furthest = 0;
                            for (index_type l = 0; l < this->edges[i].length(); ++l) {
                                neighbor = this->edges[i][l];
                                furthest_neighbor = this->edges[i][furthest];
                                
                                if (this->euclidean_distance_squared(this->vertices[i], this->vertices[furthest_neighbor]) < this->euclidean_distance_squared(this->vertices[i], this->vertices[neighbor])) {
                                    furthest = l;
                                }
                            }
                            distance_furthest = this->euclidean_distance_squared(this->vertices[i], this->vertices[this->edges[i][furthest]]);
                        }
                    } else {
                        if (this->edges[i].length() == 0) {
                            distance_furthest = dist;
                        } else {
                            distance_furthest = std::max(dist, distance_furthest);
                        }
                        this->edges[i].push_back(j);
                    } 
                }
            }
            
        }
        this->edges_number = n * this->k;
        std::cout << "Exact " << this->k << "-NNGraph built." << std::endl;
    }
};

template <typename T = double>
std::ostream& operator<<(std::ostream &out, const typename KNN_Graph<T>::Adjacency_List &a) {
    const auto size = a.size();
    for (auto i = size-size; i < size-1; ++i) {
        out << a.get(i);
        out << ", ";
    }
    out << a.get(size-1) << std::endl;
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream &out, const KNN_Graph<T> &g) {
    const auto size = g.number_vertices();
    for (auto i = size-size; i < size; ++i) {
        out << i << "-> ";
        out << g.get_neighbors(i);
    }
    return out;
}
