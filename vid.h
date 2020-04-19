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

// Struct of numbers for given processor
struct NumStruct
{
    float originAltitude;
    uint32_t globalOffset;
    uint32_t globalCount;
    Numbers numbers;

    // Converts altitudes to angles 
    void toAngles();
};

// Offsets for scattering and gathering misaligned numbers
struct Offsets
{
    std::vector<int> counts;
    std::vector<int> displacements;
};

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
    void setWorldSize(int size);

private:
    Offsets mOffsets;
    int mRank;
    int mWorldSize;
};

// Parse given input
Numbers parseInput(std::string_view input);

// Generates given count of numbers
Numbers generateNumbers(size_t count);

// Scatters numbers across processes
NumStruct scatterNumbers(Process& process, Numbers& gNumbers);

// Gathers numbers from processes
void gatherNumbers(Process& process, Numbers& gNums, NumStruct& lNums);

// Performs prefix max scan on given numbers
Numbers prefixScan(Process& process, NumStruct& nums);

// Performs visibility algorithm
void visibility(Process& process, Numbers& gNums);
