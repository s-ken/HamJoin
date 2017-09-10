#ifndef INCLUDED_TYPE
#define INCLUDED_TYPE

#include <vector>
#include <boost/align/aligned_allocator.hpp>

template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

#endif
