#pragma once 
/*
 * 3. PRL project, Visibility
 *
 * author: Bc. Matej Karas
 * email: xkaras34@stud.fit.vutbr.cz
 *
 */

#include <mpi.h>
#include <vector>
#include <string_view>

using Numbers = std::vector<float>;

class Process;

struct NumStruct
{
    float originAltitude;
    uint32_t globalOffset;
    Numbers numbers;

    // Converts altitudes to angles 
    void toAngles();
};

struct Offsets
{
    std::vector<int> counts;
    std::vector<int> displacements;
};


Numbers parseInput(std::string_view input);
NumStruct scatterNumbers(Process& process, Numbers& gNumbers);
void visibility(Process& process, Numbers& gNums);

// Helper class for current MPI process
class Process
{
public:
    static const int ROOT = 0;

public:
    Process(int& argc, char**& argv);
    ~Process();

    void createOffsets(std::vector<float>& numbers);

    int& rank();
    int& worldSize();
    bool isRoot();
    Offsets& getOffests();

private:
    Offsets mOffsets;
    int mRank;
    int mWorldSize;
};
