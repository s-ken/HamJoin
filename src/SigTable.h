#ifndef INCLUDED_SIGTABLE
#define INCLUDED_SIGTABLE

#include "config.h"
#include "type.h"
#include "SetTable.h"

#include <vector>
#include <omp.h>

#if defined( __AVX512F__ )
    #include <zmmintrin.h>
    static constexpr int32_t int32_max alignas(64) [] = { INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX };
    static constexpr uint32_t mask1 alignas(64) [] = { 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555 };
    static constexpr uint32_t mask2 alignas(64) [] = { 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333 };
    static constexpr uint32_t mask3 alignas(64) [] = { 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F };
    static constexpr uint32_t mask4 alignas(64) [] = { 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF };
    static constexpr uint32_t mask5 alignas(64) [] = { 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF };
    inline uint32_t SIMD_popcnt( __m512i vec )
    {
        vec = _mm512_sub_epi32( vec, _mm512_and_epi32( _mm512_load_epi32( mask1 ), _mm512_srli_epi32( vec, 1 ) ) );
        vec = _mm512_add_epi32( _mm512_and_epi32( _mm512_load_epi32( mask2 ), vec ), _mm512_and_epi32( _mm512_load_epi32( mask2 ), _mm512_srli_epi32( vec, 2 ) ) );
        vec = _mm512_and_epi32( _mm512_load_epi32( mask3 ), _mm512_add_epi32( vec, _mm512_srli_epi32( vec, 4) ) );
        vec = _mm512_and_epi32( _mm512_load_epi32( mask4 ), _mm512_add_epi32( vec, _mm512_srli_epi32( vec, 8) ) );
        vec = _mm512_and_epi32( _mm512_load_epi32( mask5 ), _mm512_add_epi32( vec, _mm512_srli_epi32( vec, 16) ) );
        return _mm512_reduce_add_epi32( vec );
    }
#elif defined ( __AVX2__ )
    #include <x86intrin.h>
    static constexpr int32_t int32_max alignas(64) [] = { INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX };
    static constexpr uint32_t mask1 alignas(64) [] = { 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555, 0x55555555 };
    static constexpr uint32_t mask2 alignas(64) [] = { 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333, 0x33333333 };
    static constexpr uint32_t mask3 alignas(64) [] = { 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F, 0x0F0F0F0F };
    static constexpr uint32_t mask4 alignas(64) [] = { 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF };
    static constexpr uint32_t mask5 alignas(64) [] = { 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF, 0x0000FFFF };
    inline uint32_t SIMD_popcnt( __m256i vec )
    {
        vec = _mm256_sub_epi32( vec, _mm256_and_si256( _mm256_load_si256( (__m256i *)mask1 ), _mm256_srli_epi32( vec, 1 ) ) );
        vec = _mm256_add_epi32( _mm256_and_si256( _mm256_load_si256( (__m256i *)mask2 ), vec ), _mm256_and_si256( _mm256_load_si256( (__m256i *)mask2 ), _mm256_srli_epi32( vec, 2 ) ) );
        vec = _mm256_and_si256( _mm256_load_si256( (__m256i *)mask3 ), _mm256_add_epi32( vec, _mm256_srli_epi32( vec, 4 ) ) );
        vec = _mm256_and_si256( _mm256_load_si256( (__m256i *)mask4 ), _mm256_add_epi32( vec, _mm256_srli_epi32( vec, 8 ) ) );
        vec = _mm256_and_si256( _mm256_load_si256( (__m256i *)mask5 ), _mm256_add_epi32( vec, _mm256_srli_epi32( vec, 16 ) ) );

        vec = _mm256_add_epi32( vec, _mm256_permutevar8x32_epi32( vec, _mm256_set_epi32(6,7,4,5,2,3,0,1) ) );
        vec = _mm256_add_epi32( vec, _mm256_permutevar8x32_epi32( vec, _mm256_set_epi32(4,5,6,7,0,1,2,3) ) );
        vec = _mm256_add_epi32( vec, _mm256_permutevar8x32_epi32( vec, _mm256_set_epi32(0,1,2,3,4,5,6,7) ) );
        return ((int32_t *)&vec)[0];
    }
#endif

class SigTable {
    using frag = uint32_t;
private:
    static constexpr int K = 512;
    static constexpr int NUM_BYTES_FRAG = 4;
    static constexpr int K_PER_FRAG = NUM_BYTES_FRAG * 8;
    static constexpr int NUM_FRAGS_PER_SIG = K / K_PER_FRAG;
    static constexpr float THRESHOLD_HAMMING = (1. - THRESHOLD) / 2 * K;
    size_t m_size;
    frag *sig;
    static std::vector< int32_t > m_salts;
    inline int32_t hash( int32_t k, const int hashID )
    {
        int32_t h = m_salts[hashID];
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
        h ^= 4;
        h = (h >> 16) ^ h;
        h *= 0x45d9f3b;
        h = (h >> 16) ^ h;
        h *= 0x45d9f3b;
        h = (h >> 16) ^ h;
        return h;
    }
#if defined( __AVX512F__ )
    static constexpr int SIMD_LOOP_COUNT = K / ( SIMD_WIDTH * 8 );
    __m512i simd_hash( __m512i vec, const int hashID )
    {
        __m512i h = _mm512_set1_epi32( m_salts[hashID] );
        k = _mm512_mullo_epi32( k, _mm512_set1_epi32(0xcc9e2d51 ) );
        k = _mm512_or_epi32( _mm512_slli_epi32( k, 15 ), _mm512_srli_epi32( k, 17 ) );
        k = _mm512_mullo_epi32( k, _mm512_set1_epi32(0x1b873593 ) );
        h = _mm512_xor_epi32( h, k );
        h = _mm512_or_epi32( _mm512_slli_epi32( h, 13 ), _mm512_srli_epi32( h, 19 ) );
        h = _mm512_add_epi32( _mm512_mullo_epi32( h, _mm512_set1_epi32( 5 ) ), _mm512_set1_epi32( 0xe6546b64 ) );
        h = _mm512_xor_epi32( h, _mm512_set1_epi32( 4 ) );
        h = _mm512_xor_epi32( _mm512_srli_epi32( h, 16 ), h );
        h = _mm512_mullo_epi32( h, _mm512_set1_epi32( 0x45d9f3b ) );
        h = _mm512_xor_epi32( _mm512_srli_epi32( h, 16 ), h );
        h = _mm512_mullo_epi32( h, _mm512_set1_epi32( 0x45d9f3b ) );
        h = _mm512_xor_epi32( _mm512_srli_epi32( h, 16 ), h );
        return h;
    }
#elif defined( __AVX2__ )
    static constexpr int SIMD_LOOP_COUNT = K / ( SIMD_WIDTH * 8 );
    __m256i simd_hash( __m256i k, const int hashID )
    {
        __m256i h = _mm256_set1_epi32( m_salts[hashID] );
        k = _mm256_mullo_epi32( k, _mm256_set1_epi32( 0xcc9e2d51 ) );
        k = _mm256_or_si256( _mm256_slli_epi32( k, 15 ), _mm256_srli_epi32( k, 17 ) );
        k = _mm256_mullo_epi32( k, _mm256_set1_epi32( 0x1b873593 ) );
        h = _mm256_xor_si256( h, k );
        h = _mm256_or_si256( _mm256_slli_epi32( h, 13 ), _mm256_srli_epi32( h, 19 ) );
        h = _mm256_add_epi32( _mm256_mullo_epi32( h, _mm256_set1_epi32( 5 ) ), _mm256_set1_epi32( 0xe6546b64 ) );
        h = _mm256_xor_si256( h, _mm256_set1_epi32( 4 ) );
        h = _mm256_xor_si256( _mm256_srli_epi32( h, 16 ), h );
        h = _mm256_mullo_epi32( h, _mm256_set1_epi32( 0x45d9f3b ) );
        h = _mm256_xor_si256( _mm256_srli_epi32( h, 16 ), h );
        h = _mm256_mullo_epi32( h, _mm256_set1_epi32( 0x45d9f3b ) );
        h = _mm256_xor_si256( _mm256_srli_epi32( h, 16 ), h );
        return h;
    }
#endif

public:
    SigTable()
    {
        if ( m_salts.empty() ) {
            std::srand( 149 );
            m_salts.resize( K );
            for( auto & salt : m_salts ) {
                salt = (int32_t)(std::rand());
            }
        }
    }

    ~SigTable()
    {
        _mm_free( sig );
    }
    inline size_t size()
    {
        return m_size;
    }
    inline size_t head_pos( int i_rec )
    {
        return i_rec * NUM_FRAGS_PER_SIG;
    }
    inline size_t len() {
        return NUM_FRAGS_PER_SIG;
    }
    inline frag *head( int i_rec ) {
        return sig + head_pos( i_rec );
    }
    frag * operator [](int n)
    {
        return head( n );
    }
    void print_info()
    {
        int sz_b = K * size() / 8;
        double sz_kb = sz_b / 1024;
        double sz_mb = sz_kb / 1024;
        std::cout << "Size of Signature Table = " << sz_mb << " MB" << std::endl;
    }
    void sketch( SetTable & );
    void join( std::vector< std::pair< int, int > >& );
};

std::vector< int32_t > SigTable::m_salts;

void SigTable::sketch( SetTable &setTable )
{
    m_size = setTable.size();
    sig = (frag *)_mm_malloc( NUM_FRAGS_PER_SIG * sizeof(frag) * m_size, 64 );
    memset( (void *)sig, 0, NUM_FRAGS_PER_SIG * sizeof(frag) * m_size );

    // b-bit MinHash
    #pragma omp parallel for num_threads( NUM_THREADS )
    for( int i_rec = 0; i_rec < m_size; i_rec++ ) {

        // Culclate the signature of (i_rec)-th record
        for( int k = 0; k < K; k++ ) {

            int32_t minVal;
#if defined( __AVX512F__ )
            __m512i minVals = _mm512_load_epi32( int32_max );
            for(int p = 0; p < setTable.len(i_rec); p += SIMD_WIDTH_INT) {
                minVals = _mm512_min_epi32( minVals, simd_hash( _mm512_load_epi32( setTable.head(i_rec)+p ), k ) );
            }
            minVal = _mm512_reduce_min_epi32( minVals );
#elif defined ( __AVX2__ )
            __m256i minVals = _mm256_load_si256( (__m256i *)int32_max );
            for(int p = 0; p < setTable.len(i_rec); p += SIMD_WIDTH_INT) {
                minVals = _mm256_min_epi32( minVals, simd_hash( _mm256_load_si256( (__m256i *)( &setTable[i_rec][p] ) ), k ) );
            }
            minVals = _mm256_min_epi32( minVals, _mm256_permutevar8x32_epi32( minVals, _mm256_set_epi32(6,7,4,5,2,3,0,1) ) );
            minVals = _mm256_min_epi32( minVals, _mm256_permutevar8x32_epi32( minVals, _mm256_set_epi32(4,5,6,7,0,1,2,3) ) );
            minVals = _mm256_min_epi32( minVals, _mm256_permutevar8x32_epi32( minVals, _mm256_set_epi32(0,1,2,3,4,5,6,7) ) );
            minVal = ((int32_t *)&minVals)[0];
#else
            minVal = INT32_MAX;
            for(int p = 0; p < setTable.len(i_rec); p++) {
                minVal = std::min( minVal, hash( setTable[i_rec][p], k ) );
            }
#endif
            head( i_rec )[ k / K_PER_FRAG ] |= ( ( minVal & 0x00000001 ) << ( k % K_PER_FRAG ) );
        }
    }
}

void SigTable::join( std::vector< std::pair< int, int > >& dst )
{
    dst.clear();
    std::vector< std::vector< std::pair< int32_t,int32_t> > > dst_local( NUM_THREADS );
    std::vector< int > size_local( NUM_THREADS );
    std::vector< int > size_local_scan( NUM_THREADS );

    #pragma omp parallel shared( dst_local, size_local, size_local_scan ) num_threads( NUM_THREADS )
    {
        const int tid = omp_get_thread_num();
        size_local[tid] = ( m_size + tid ) / NUM_THREADS;
        #pragma omp barrier
        #pragma omp single
        {
            // Compute the prefix-sum of size_local to identify head-place of each thread
            for ( int i = 1; i < NUM_THREADS; i++ ) {
                size_local_scan[i] = size_local_scan[i-1] + size_local[i-1];
            }
        }
        const int begin_local = size_local_scan[tid];
        const int end_local = begin_local + size_local[tid];

        // Perform join
        for ( int i_rec = begin_local; i_rec < end_local; i_rec++ ) {
#if defined( __AVX512F__ )
            __m512i r[SIMD_LOOP_COUNT];
            #pragma unroll
            for ( int i = 0; i < SIMD_LOOP_COUNT; i++ ) {
                r[i] = _mm512_load_epi32( head( i_rec ) + i * SIMD_WIDTH_INT );
            }
#elif defined( __AVX2__ )
            __m256i r[SIMD_LOOP_COUNT];
            for ( int i = 0; i < SIMD_LOOP_COUNT; i++ ) {
                r[i] = _mm256_load_si256( (__m256i *)( head( i_rec ) + i * SIMD_WIDTH_INT ) );
            }
#endif
            for ( int j_rec = 0; j_rec < m_size; j_rec++ ) {
#if defined( __AVX512F__ )
                int sum = 0;
                #pragma unroll
                for ( int i = 0; i < SIMD_LOOP_COUNT; i++ ) {
                    sum += SIMD_popcnt( _mm512_xor_epi32( r[i], _mm512_load_epi32( head( j_rec ) + i * SIMD_WIDTH_INT ) ) );
                }
#elif defined( __AVX2__ )
                int sum = 0;
                for ( int i = 0; i < SIMD_LOOP_COUNT; i++ ) {
                    sum += SIMD_popcnt( _mm256_xor_si256( r[i], _mm256_load_si256( (__m256i *)( head( j_rec ) + i * SIMD_WIDTH_INT ) ) ) );
                }
#else
                int sum = 0;
                for ( int i = 0; i < NUM_FRAGS_PER_SIG; i++ ) {
                    sum += _popcnt32( head( i_rec )[i] ^ head( j_rec )[i] );
                }
#endif
                if ( sum <= THRESHOLD_HAMMING ) {
                    dst_local[tid].push_back( std::make_pair( i_rec, j_rec ) );
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
