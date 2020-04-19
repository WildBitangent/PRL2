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
		numbers.emplace_back(std::strtof(input.begin() + prev, nullptr));
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
	result.globalCount = count;

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
	size_t maxStride = 0;
	
	// 1st stage - up-sweep locally 
	{
		size_t start = 0;
	
		for (size_t stride = 1; stride < sums.size(); stride <<= 1)
		{
			size_t next = start;
			for (; next + stride < sums.size(); next += 2 * stride)
				sums[next + stride] = std::max(sums[next], sums[next + stride]);

			// unaligned data (tree) correction
			if (next + 1 < sums.size())
				sums.back() = std::max(sums[next], sums.back());

			start += stride;
			maxStride = stride;
		}
	}

	auto upsweepMax = [&process, &sums](size_t sendId, size_t recvId)
	{
		int tag = sendId | (recvId << 8);
		float recv;

		if (process.rank() == sendId)
			MPI_Send(&sums.back(), 1, MPI_FLOAT, recvId, tag, MPI_COMM_WORLD);
		
		if (process.rank() == recvId)
		{
			MPI_Recv(&recv, 1, MPI_FLOAT, sendId, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sums.back() = std::max(sums.back(), recv);
		}
	};

	// 2nd stage - parallel up-sweep and down-sweep
	{
		// up-sweep
		size_t start = 0;
		size_t maxGlobStride = 1;

		for (size_t stride = 1; stride < process.worldSize(); stride <<= 1)
		{
			size_t next = start;
			for (; next + stride < process.worldSize(); next += 2 * stride)
				upsweepMax(next, next + stride);

			// unaligned data (tree) correction
			if (next + 1 < process.worldSize())
				upsweepMax(next, process.worldSize() - 1);

			start += stride;
			maxGlobStride = stride;
		}

		auto downsweepMax = [&process, &sums](size_t sendId, size_t recvId)
		{
			if (process.rank() == sendId || process.rank() == recvId)
			{	
				int sendtag = sendId | (recvId << 8);
				float recv;

				if (process.rank() != sendId)
					std::swap(sendId, recvId);
				
				MPI_Sendrecv(
					&sums.back(), 1, MPI_FLOAT, recvId, sendtag, 
					&recv, 1, MPI_FLOAT, recvId, sendtag, 
					MPI_COMM_WORLD, MPI_STATUS_IGNORE
				);

				if (process.rank() < std::max(sendId, recvId)) // left child
					sums.back() = recv;
				else // right child
					sums.back() = std::max(sums.back(), recv);
			}
		};

		//down-sweep
		if (process.rank() == process.worldSize() - 1)
			sums.back() = -std::numeric_limits<float>::max();

		for (size_t stride = maxGlobStride; stride > 0; stride >>= 1)
		{
			size_t start = stride - 1;
			size_t next = start;

			for (; next + stride < process.worldSize(); next += 2 * stride);			

			// unaligned data (tree) correction
			if (next + 1 < process.worldSize())
				downsweepMax(next, process.worldSize() - 1);

			for (size_t i = start; i + stride < process.worldSize(); i += 2 * stride)
				downsweepMax(i, i + stride);
		}
	}

	// 3rd stage - down-sweep locally on every core in parallel
	{
		for (size_t stride = maxStride; stride > 0; stride >>= 1)
		{
			size_t start = stride - 1;
			size_t next = start;

			for (; next + stride < sums.size(); next += 2 * stride);

			// unaligned data (tree) correction
			if (next + 1 < sums.size())
			{
				std::swap(sums[next], sums.back());
				sums.back() = std::max(sums[next], sums.back());
			}

			for (size_t i = start; i + stride < sums.size(); i += 2 * stride)
			{	
				std::swap(sums[i], sums[i + stride]);
				sums[i + stride] = std::max(sums[i], sums[i + stride]);
			}
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

	if (numStruct.globalCount == 0)
	{
		if (process.isRoot())
			gNums.front() = '_';
		return;
	}

	auto prevWorldSize = process.worldSize();
	if (process.worldSize() > numStruct.globalCount)
		process.setWorldSize(numStruct.globalCount);

	#if BENCHMARK
		auto scatterPoint = clk::now();
		auto scatterTime = duration_cast<std::chrono::duration<float, std::milli>>(scatterPoint - start).count();
		if (process.isRoot())
			std::cout << "Scatter time: " << scatterTime << " ms" <<  std::endl;
	#endif

	// parallel to angles
	numStruct.toAngles();

	// parallel prefix max scan
	auto maxPrescan = prefixScan(process, numStruct);

	// decide if point is visible (this approach probabli ain't best idea, if cluster nodes has different endianess)
	for (size_t i = 0; i < numStruct.numbers.size(); ++i)
		numStruct.numbers[i] = (numStruct.numbers[i] > maxPrescan[i] ? 'v' : 'u');

	if (process.isRoot())
		gNums.front() = '_';

	#if BENCHMARK
		auto algPoint = clk::now();
		auto algTime = duration_cast<std::chrono::duration<float, std::milli>>(algPoint - scatterPoint).count();
		if (process.isRoot())
			std::cout << "Algorithm time: " << algTime << " ms" <<  std::endl;
	#endif
		
	process.setWorldSize(prevWorldSize);
	gatherNumbers(process, gNums, numStruct);

	#if BENCHMARK
		auto gatherPoint = clk::now();
		auto gatherTime = duration_cast<std::chrono::duration<float, std::milli>>(gatherPoint - algPoint).count();
		auto time = duration_cast<std::chrono::duration<float, std::milli>>(gatherPoint - start).count();
		
		if (process.isRoot())
		{
			std::cout << "Gather Time: " << gatherTime << " ms" << std::endl;
			std::cout << "Total Time: " << time << " ms" << std::endl;
		}
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

void Process::setWorldSize(int size)
{
	mWorldSize = size;
}