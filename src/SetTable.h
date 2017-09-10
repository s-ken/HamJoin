#ifndef INCLUDED_SETTABLE
#define INCLUDED_SETTABLE

#include "config.h"
#include "type.h"

#include <set>
#include <map>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include <iostream>

static std::set< std::string > stop_words{"a","able","about","across","after","al","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","e","either","else","et","ever","every","for","from","g","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"};
static std::string delims = " ,./;:\"\'()[]{}<>&!~_";

static bool is_stop_word ( const std::string& word )
{
    if ( stop_words.find(word) != stop_words.end() || all_of( word.cbegin(), word.cend(), ::isdigit ) ) {
        return true;
    }
    return false;
}

class SetTable
{
private:
    aligned_vector< int32_t > m_end, m_tok;
    static constexpr int ALIGNMENT_WIDTH = 16;
    std::map< std::string, int > m_wordIDs;
    void construct( std::string );
    inline size_t head_pos(const int i_rec)
    {
        return (m_end[i_rec] + ALIGNMENT_WIDTH - 1) / ALIGNMENT_WIDTH * ALIGNMENT_WIDTH;
    }
    inline size_t tail_pos(const int i_rec)
    {
        return m_end[i_rec+1];
    }

public:
    SetTable(std::string name)
    {
        construct( name );
    }
    ~SetTable() {}
    inline size_t size()
    {
        return m_end.size() - 1;
    }
    inline size_t len(const int i_rec)
    {
        return tail_pos(i_rec) - head_pos(i_rec);
    }
    inline int32_t *head(const int i_rec)
    {
        return &m_tok[head_pos(i_rec)];
    }
    int32_t * operator [](int n)
    {
        return head( n );
    }
    /*
    void print()
    {
        // for debugging
        for ( int i_rec = 0; i_rec < size(); i_rec++ ) {
            for ( int p = 0; p < len(i_rec); p += SIMD_WIDTH_INT ) {
                for ( int i = 0; i < SIMD_WIDTH_INT; i++ ) {
                    std::cout << head(i_rec)[p+i] << ", " << std::flush;
                }
            }
            std::cout << std::endl;
        }
    }
    */
    void join( std::vector< std::pair< int, int > >& );
};

void SetTable::construct( std::string name )
{
    std::ifstream ifs(name);
    if ( ifs.fail() ) {
        std::cerr << "Error: Opening file \"" << name << "\"" << std::endl;
        std::exit( EXIT_FAILURE );
    }

    m_end.push_back( 0 );
    std::string line;
    while ( std::getline( ifs, line ) ) {
        std::list< std::string > words;

        // Get words of a record and clean it.
        std::transform( line.begin(), line.end(), line.begin(), ::tolower );
        boost::split( words, line, boost::is_any_of( delims ) );
        words.sort();
        words.unique();
        words.remove_if( is_stop_word );

        // Insert a record to m_tok and m_end
        for ( auto & word : words ) {
            int wordID;
            auto itr = m_wordIDs.find(word);
            if ( itr == m_wordIDs.end() ) {
                wordID = m_wordIDs.size();
                m_wordIDs[word] = wordID;
            } else {
                wordID = itr -> second;
            }
            m_tok.push_back( wordID );
        }
        m_end.push_back( m_tok.size() );

        // In order to align head of each record on the boundary, m_tok are padded by its last element
        for ( int i = m_tok.size(); i % ALIGNMENT_WIDTH != 0; i++ ) {
            m_tok.push_back( m_tok.back() );
        }
    }
}

void SetTable::join( std::vector< std::pair< int, int > >& dst )
{
    dst.clear();
    int num_threads = omp_get_max_threads();
    std::vector< std::vector< std::pair< int32_t,int32_t> > > dst_local( num_threads );
    std::vector< int > size_local( num_threads );
    std::vector< int > size_local_scan( num_threads );

    #pragma omp parallel shared( dst_local, size_local, size_local_scan ) num_threads( NUM_THREADS )
    {
        const int tid = omp_get_thread_num();
        size_local[tid] = ( size() + tid ) / num_threads;
        #pragma omp barrier
        #pragma omp single
        {
            // Compute the prefix-sum of size_local to identify head-place of each thread
            for ( int i = 1; i < num_threads; i++ ) {
                size_local_scan[i] = size_local_scan[i-1] + size_local[i-1];
            }
        }
        const int begin_local = size_local_scan[tid];
        const int end_local = begin_local + size_local[tid];

        // Perform join
        for ( int i_rec = begin_local; i_rec < end_local; i_rec++ ) {
            std::unordered_map< int32_t, int > hash_table;
            for ( int i = 0; i < len( i_rec ); i++ ) hash_table[ head(i_rec)[i] ] = 1;
            for ( int j_rec = 0; j_rec < size(); j_rec++ ) {
                int intersection = 0;
                for ( int i = 0; i < len( j_rec ); i++ ) {
                    if ( hash_table.find( head(j_rec)[i] ) != hash_table.end() ) intersection++;
                }
                if ( (float)intersection / ( len(i_rec) + len(j_rec) - intersection ) >= THRESHOLD ) {
                    dst.push_back( std::make_pair( i_rec, j_rec ) );
                }
            }
        }
    }

    // Copy the results to dst
    int size_dst = 0;
    for ( auto & v : dst_local ) size_dst += v.size();
    dst.reserve( size_dst );
    for ( auto & v : dst_local ) std::copy( v.begin(), v.end(), std::back_inserter( dst ) );
}

#endif
