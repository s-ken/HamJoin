#ifndef INCLUDED_FUNCTIONS
#define INCLUDED_FUNCTIONS

#include <utility>
#include <vector>
#include <unordered_map>
#include <functional>
#include <boost/unordered_map.hpp>

template<typename T> constexpr
T const& max(T const& a, T const& b) {
  return a > b ? a : b;
}

template<typename T> constexpr
T const& min(T const& a, T const& b) {
  return a < b ? a : b;
}

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () ( const std::pair< T1, T2 > & p ) const
    {
        std::size_t seed = 149;
        boost::hash_combine(seed, p.first);
        boost::hash_combine(seed, p.second);
        return seed;
    }
};
std::pair< double, double > eval_recall_precision( std::vector< std::pair< int, int > >& correct_results, std::vector< std::pair< int, int > >& results )
{
    int TP = 0, FP, FN;
    std::unordered_map< std::pair< int, int >, int, pair_hash > hash_table;
    for ( auto & correct_result : correct_results ) hash_table[ correct_result ] = 1;
    for ( auto & result : results ) {
        if ( hash_table.find( result ) != hash_table.end() ) TP++;
    }
    FP = results.size() - TP;
    FN = correct_results.size() - TP;

    return std::make_pair( (double)TP/(TP+FP), (double)TP/(TP+FN) ); // ( Precision, Recall )
}

#endif

