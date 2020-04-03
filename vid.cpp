/*
 * 3. PRL project, Visibility
 *
 * author: Bc. Matej Karas
 * email: xkaras34@stud.fit.vutbr.cz
 *
 */

#include "vid.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iterator>
#include <chrono>

using namespace std::chrono;
using clk = high_resolution_clock;

int main(int argc, char *argv[])
{
	Process process(argc, argv);

	Numbers globalNumbers;
	if (process.isRoot())
	{
		#if BENCHMARK
			globalNumbers = generateNumbers(std::strtol(argv[1], nullptr, 10));
		#else
			globalNumbers = parseInput(argv[1]);
		#endif
		process.createOffsets(globalNumbers);
	}

	visibility(process, globalNumbers);

	// print numbers
	#if !BENCHMARK
		if (process.isRoot())
		{
			for (size_t i = 0; i < globalNumbers.size() - 1; ++i)
				std::cout << static_cast<char>(globalNumbers[i]) << ",";
			std::cout << static_cast<char>(globalNumbers.back()) << std::endl;
		}
	#endif

	return EXIT_SUCCESS;
}

////////////////////////////////////////////////
/// Functions
////////////////////////////////////////////////
Numbers parseInput(std::string_view input)
{
	std::vector<float> numbers;
	size_t prev = 0;
	size_t pos;

	// parse comma separated values
	do
	{
		pos = input.find(",", prev);
		numbers.emplace_back(std::strtof(input.begin() + prev, nullptr)); // TODO maybe handle errno
		prev = pos + 1;
	} while (pos != std::string::npos);

	return numbers;
}

Numbers generateNumbers(size_t count)
{
	std::cout << "Nums generated: " << count << std::endl;
	std::vector<float> data;
	for (size_t i = 0; i < count; ++i)
		data.emplace_back(rand() % 1000);

	return data;
}

NumStruct scatterNumbers(Process& process, Numbers& numbers)
{	
	NumStruct result;
	auto& sendCounts = process.getOffests().counts;
	auto& displacements = process.getOffests().displacements;
	
	// broadcast how many numbers is incoming
	int count = numbers.size() - 1;
	MPI_Bcast(&count, 1, MPI_INT, Process::ROOT, MPI_COMM_WORLD);

	auto getCount = [&count, &process](int rank) {
		return count / process.worldSize() + (rank < count % process.worldSize());
	};

	if (process.isRoot())
		result.originAltitude = numbers[0];

	// broadcast origin altitude
	MPI_Bcast(&result.originAltitude, 1, MPI_FLOAT, Process::ROOT, MPI_COMM_WORLD);

	// Scatter global offsets
	MPI_Scatter(displacements.data(), 1, MPI_UINT32_T, &result.globalOffset, 1, MPI_UINT32_T, Process::ROOT, MPI_COMM_WORLD);

	// set recv buffer size
	count = getCount(process.rank());
	result.numbers.resize(count); 

	// Scatter data across processes
	MPI_Scatterv(
		numbers.data() + 1, 
		sendCounts.data(), 
		displacements.data(), 
		MPI_FLOAT, 
		&result.numbers[0], 
		count, 
		MPI_FLOAT, 
		Process::ROOT, 
		MPI_COMM_WORLD
	);
	return result;
}

void gatherNumbers(Process& process, Numbers& gNums, NumStruct& lNums)
{
	MPI_Gatherv(
		lNums.numbers.data(), 
		lNums.numbers.size(), 
		MPI_FLOAT, 
		gNums.data() + 1, 
		process.getOffests().counts.data(),
		process.getOffests().displacements.data(), 
		MPI_FLOAT, 
		Process::ROOT,
		MPI_COMM_WORLD
	);
}

Numbers prefixScan(Process& process, NumStruct& nums)
{
	Numbers sums(nums.numbers.begin(), nums.numbers.end());
	sums.reserve(1); // fix when num count < processor count

	// 1st stage - set maxes on every core in parallel
	{
		size_t start = 0;
		for (size_t stride = 1; stride < std::ceil(std::log2(sums.size())); stride <<= 1)
		{
			size_t prevPos = start;
			for (size_t i = start; i + stride < sums.size(); prevPos = i, i += 2 * stride)
				sums[i + stride] = std::max(sums[i], sums[i + stride]);

			// unaligned data (tree) correction
			if (prevPos + stride >= sums.size())
				sums.back() = std::max(sums[prevPos], sums.back());

			start += stride;
		}
	}

	float gMax = -std::numeric_limits<float>::max();

	// 2nd stage - maxes from cores to consecutive ranks
	{
		if (process.isRoot())
			MPI_Send(&sums.back(), 1, MPI_FLOAT, process.rank() + 1, 42, MPI_COMM_WORLD);
		else
		{
			MPI_Recv(&gMax, 1, MPI_FLOAT, process.rank() - 1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			if (process.rank() != process.worldSize() - 1)
			{
				float max = std::max(sums.back(), gMax);
				MPI_Send(&max, 1, MPI_FLOAT, process.rank() + 1, 42, MPI_COMM_WORLD);
			}
		}
	}

	// 3rd stage - finish scan on every core in parallel (make prefix, from partial infix)
	{
		for (auto &e : sums)
		{
			std::swap(gMax, e);
			gMax = std::max(gMax, e);
		}
	}

	return sums;
}

void visibility(Process& process, Numbers& gNums)
{
	#if BENCHMARK
		auto start = clk::now();
	#endif

	auto numStruct = scatterNumbers(process, gNums);

	// parallel to angles
	numStruct.toAngles();

	// parallel prefix max scan
	auto maxPrescan = prefixScan(process, numStruct);

	// decide if point is visible (this approach probabli ain't best idea, if cluster nodes has different endianess)
	for (size_t i = 0; i < numStruct.numbers.size(); ++i)
		numStruct.numbers[i] = (numStruct.numbers[i] > maxPrescan[i] ? 'v' : 'u');

	if (process.isRoot())
		gNums.front() = '_';
		
	gatherNumbers(process, gNums, numStruct);

	#if BENCHMARK
		auto time = duration_cast<std::chrono::duration<float, std::milli>>(clk::now() - start).count();
		if (process.isRoot())
			std::cout << "Algorithm Time: " << time << " ms" << std::endl;
	#endif
}

////////////////////////////////////////////////
/// NumStruct
////////////////////////////////////////////////
void NumStruct::toAngles()
{
	for (size_t i = 0; i < numbers.size(); ++i)
		numbers[i] = std::atan((numbers[i] - originAltitude) / (i + globalOffset + 1));
}

////////////////////////////////////////////////
/// Process class
////////////////////////////////////////////////
Process::Process(int& argc, char**& argv)
{
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mWorldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
}

Process::~Process()
{
	MPI_Finalize();
}

void Process::createOffsets(std::vector<float>& numbers)
{
	size_t displacement = 0;
	size_t count = numbers.size() - 1;
	mOffsets.counts.resize(mWorldSize);
	mOffsets.displacements.resize(mWorldSize);

	for (size_t i = 0; i < mWorldSize; ++i)
	{
		mOffsets.counts[i] = count / mWorldSize + (i < count % mWorldSize);
		mOffsets.displacements[i] = displacement;
		displacement += mOffsets.counts[i];
	}
}

int& Process::rank()
{
	return mRank;
}

int& Process::worldSize()
{
	return mWorldSize;
}

bool Process::isRoot()
{
	return mRank == ROOT;
}

Offsets& Process::getOffests()
{
	return mOffsets;
}
