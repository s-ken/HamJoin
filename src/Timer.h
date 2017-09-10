#ifndef INCLUDED_TIME
#define INCLUDED_TIME

#include <iostream>
#include <chrono>

class Timer
{
private:
    std::chrono::system_clock::time_point time_stamp;
    double duration;
    int count;

public:
    Timer()
    {
        reset();
    }
    void reset()
    {
        duration = 0;
        count = 0;
    }
    void start()
    {
        time_stamp = std::chrono::system_clock::now();
    }
    void stop()
    {
        std::chrono::system_clock::time_point t = std::chrono::system_clock::now();
        duration += std::chrono::duration_cast< std::chrono::milliseconds >( t-time_stamp ).count();
        count++;
    }
    void print()
    {
        std::cout << "Elapsed Time [ms] = " << duration / count << std::endl;
    }
    void end()
    {
        stop();
        print();
        reset();
    }
};

#endif
