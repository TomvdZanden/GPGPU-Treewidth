// The following (strange) definitions are to get the indexing for shared (OpenCL: local) arrays correct.
// Best performance is if sections of an array allocated to a thread are not consequitive, but consquitive locations accessed by a single thread are spaced by 128 bytes (32x4 bytes)
#define cbi(x,y) (((x)[indexInt((y) >> 5)]) &= ~(1u << ((y) & 31)))
#define sbi(x,y) (((x)[indexInt((y) >> 5)]) |= (1u << ((y) & 31)))
#define tbi(x,y) (((x)[indexInt((y) >> 5)]) &  (1u << ((y) & 31)))
#define tbi_norm(x,y) (((x)[(y) >> 5]) &  (1u << ((y) & 31)))
#define sbi_norm(x,y) (((x)[(y) >> 5]) |= (1u << ((y) & 31)))
#ifdef useLocalMemory
#define ucharLocalArray(name, count) __local uchar shared_##name[(count) * localWorkSize]; __local uchar* name = shared_##name + (get_local_id(0) % 32) * 4 + (get_local_id(0) / 32) * 32 * (count);
#define uintLocalArray(name, count)  __local uint shared_##name[(count) * localWorkSize];   __local uint* name = shared_##name + (get_local_id(0) % 32) + (get_local_id(0) / 32) * 32 * (count);
#else
#define ucharLocalArray(name, count) __private uchar name[count];
#define uintLocalArray(name, count)  __private uint name[count];
#endif

#ifdef useLocalMemory
	#define indexByte(x) ((((x) >> 2) << 7) + ((x) & 3))
	#define indexInt(x) ((x) << 5)
#else
	#define indexByte(x) (x)
	#define indexInt(x) (x)
#endif

#ifdef useLocalMemory
inline uchar Find(__local uchar* rep, int x)
#else
inline uchar Find(__private uchar* rep, int x)
#endif
{
    int y = x;
    while (y != rep[indexByte(y)])
        y = rep[indexByte(y)];

    while (x != rep[indexByte(x)])
    {
        int z = rep[indexByte(x)];
        rep[indexByte(x)] = y;
        x = z;
    }

    return y;
}

#ifdef useLocalMemory
inline void Union(__local uchar* rep, int x, int y)
#else
inline void Union(__private uchar* rep, int x, int y)
#endif
{
    rep[indexByte(Find(rep, x))] = Find(rep, y);
}

// Simplified version of Murmur3 (our key always has a length that is a multiple of 4 bytes)
#ifdef useLocalMemory
inline uint Murmur3(__local uint* key, uint seed)
#else
inline uint Murmur3(__private uint* key, uint seed)
#endif
{
    uint h = seed;

	#pragma unroll
    for(int i = 0; i < numBitsetWords; i++) {
        uint k = key[indexInt(i)];
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h += (h << 2) + 0xe6546b64;
   }

   h ^= numBitsetWords << 2;
   h ^= h >> 16;
   h *= 0x85ebca6b;
   h ^= h >> 13;
   h *= 0xc2b2ae35;
   h ^= h >> 16;

   return h;
}

__kernel void DPAlgoIteration(int n,              // 0 NodeCount of graph
	int k,                                        // 1 Target value treewidth
	int In,                                       // 2 Number of input values
	volatile __global int* restrict Out,          // 3 Output buffer (gets read back)
	__global const uint* restrict InStack,        // 4 Input stack
	__global uint* restrict OutStack,             // 5 Output stack
	volatile __global uint* restrict BloomFilter, // 6 Bloom filter (empty list)
	volatile __global uint* restrict Locks,       // 7 Lock table for bloom filter
	__constant const uint* restrict Last,         // 8 "Last" values (for reducing number of options)
	__constant const uchar* restrict AdjList,     // 9 Graph adjacency list
	int filterSize,                               // 10 Current bloom filter size
	int startPoint,                               // 11 Offset for start point (used to split computation into batches)
	int z)							              // 12 Number of nodes eliminated so far
{
	int myId = get_global_id(0) + startPoint;
	
	if (myId >= In) return;

	// Set up BFS queue
	int localQueueIn = 0, localQueueOut = 0;
	ucharLocalArray(localQueue, localQueueSize);

	// Retrieve "my" job
	uintLocalArray(input, numBitsetWords + 1);
	for (int i = 0; i <= numBitsetWords; i++) input[indexInt(i)] = InStack[myId * (numBitsetWords + 1) + i];

	int curExploreNode = -1;
	int curCount = 255;
	int curNbIdx = 0;
	uintLocalArray(visited, numBitsetWords);
	uintLocalArray(elimCandidates, numBitsetWords);
	for (int i = 0; i < numBitsetWords; i++) elimCandidates[indexInt(i)] = 0;

	// For use with MMW
#ifdef useMMW
	ucharLocalArray(currentDegree, numBitsetWords * 32);
	ucharLocalArray(rep, numBitsetWords * 32);
	uintLocalArray(minDegreeVisited, numBitsetWords);
#endif

	// Compute which nodes can be feasibly eliminated
	while (true)
	{
		// Start exploring a new node
		if (localQueueIn == localQueueOut)
		{
			// curExploreNode can be eliminated at this point
#ifdef useMMW
			if (curCount <= k && !tbi_norm(Last, curExploreNode))
#else
			if (curCount <= k)
#endif
				sbi(elimCandidates, curExploreNode);

			// Computing MMW: remember degree
#ifdef useMMW
			if (curExploreNode >= 0)
				currentDegree[indexByte(curExploreNode)] = curCount;
#endif

			// Let's check the next node
			curExploreNode++;

			// We finished exploring all the nodes
			if (curExploreNode >= n) break;

#ifdef useMMW
			// For MMW: initialize reps
			rep[indexByte(curExploreNode)] = curExploreNode;
#endif

			// Check that node is not already in S, if not, put it on the queue. Also check that it is not prohibited to eliminate due to it being one of the last nodes.
#ifdef useMMW
			if (!tbi(input, curExploreNode))
#else
			if (!tbi(input, curExploreNode) && !tbi_norm(Last, curExploreNode))
#endif
			{
				localQueueIn = 0; localQueueOut = 1;
				localQueue[indexByte(0)] = curExploreNode;
				curCount = 0;
				curNbIdx = 0;
				for (int i = 0; i < numBitsetWords; i++) visited[indexInt(i)] = 0;
				sbi(visited, curExploreNode);
			}
			else
				curCount = 255;
		}

		// The queue is (now) non-empty
		for (int e = 0; e < repeat_outer && localQueueIn != localQueueOut; e++)
		{
			int curNode = localQueue[indexByte(localQueueIn)];
			int curNbCount = AdjList[curNode << maxDegreeLog];

			if (curNbIdx == curNbCount)
			{
				// We have finished exploring the current node
				curNbIdx = 0;
				localQueueIn++;
				if (localQueueIn >= localQueueSize)
					localQueueIn -= localQueueSize;
			}
			else
				for (int f = 0; f < repeat_inner && curNbIdx != curNbCount; f++)
				{
					curNbIdx++;
					int curNb = AdjList[(curNode << maxDegreeLog) + curNbIdx];

					// Check not already visited
					if (tbi(visited, curNb)) continue;

					// Mark visited
					sbi(visited, curNb);

					if (tbi(input, curNb))
					{
						// curNb is in S -> explore the path
						localQueue[indexByte(localQueueOut)] = curNb;
						localQueueOut++;
						if (localQueueOut >= localQueueSize)
							localQueueOut -= localQueueSize;
					}
					else
					{
						// curNb is not in S -> count towards the count
						curCount++;
						// If curCount becomes too high, terminate BFS
#ifndef useMMW
						if (curCount > k)
						{
							localQueueIn = localQueueOut;
							break;
						}
#endif
					}
				}
		}
	}



#ifdef useMMW
	int minDegree = 0, secondMinDegree = 0, minNbDegree = 0;
	int minNbNode = 0, minDegreeNode = 0;

	int mode = 0; // 0 == find new min degree, 1 == find smallest degree NB, 2 == compute number of common nodes

	localQueueIn = 0; localQueueOut = 0;

	int r = n - z;

	// Now let's compute MMW
	// Initially there are n - z nodes remaining in the graph
	// We are looking for an elimination order with maximum degree k
	// Any graph with at most k + 1 nodes has max degree < k
	// So we only need to consider graphs with more than k + 1 nodes
	while (r > k - 1)
	{
		while (localQueueIn == localQueueOut)
		{
			// a BFS has terminated, do something
			if (mode == 0)
			{
				// Find a minimum degree node, and start BFS to find minimum degree neighbour
				minDegree = 255; secondMinDegree = 255;
				minDegreeNode = 255;
				for (int j = 0; j < n; j++)
					if (currentDegree[indexByte(Find(rep, j))] < minDegree)
					{
						secondMinDegree = minDegree;
						minDegreeNode = Find(rep, j);
						minDegree = currentDegree[indexByte(minDegreeNode)];
					}
					else if (currentDegree[indexByte(Find(rep, j))] == minDegree)
						secondMinDegree = minDegree;

				if (secondMinDegree == 255)
					secondMinDegree = minDegree;

				if (secondMinDegree > k)
					return;

				// Do a BFS to find the neighbour with minimum degree
				minNbDegree = 255;
				minNbNode = 255;

				localQueueIn = 0; localQueueOut = 1;
				localQueue[indexByte(0)] = minDegreeNode;
				curNbIdx = 0;
				for (int j = 0; j < numBitsetWords; j++) visited[indexInt(j)] = 0;
				sbi(visited, minDegreeNode);

				mode = 1;
			}
			else if (mode == 1)
			{
				// We found the minimum degree neighbour, now start BFS to compute its adjacency list
				for (int j = 0; j < numBitsetWords; j++) minDegreeVisited[indexInt(j)] = visited[indexInt(j)];

				localQueueIn = 0; localQueueOut = 1;
				localQueue[indexByte(0)] = minNbNode;
				curNbIdx = 0;
				for (int j = 0; j < numBitsetWords; j++) visited[indexInt(j)] = 0;
				sbi(visited, minNbNode);

				mode = 2;
			}
			else
			{
				// We computed the adjacency list of minNB, now we can update the degrees and do the contraction
				for (int j = 0; j < n; j++)
				{
					if (tbi(visited, j)) sbi(visited, Find(rep, j));
					if (tbi(minDegreeVisited, j)) sbi(minDegreeVisited, Find(rep, j));
				}

				int commonNB = 0;

				for (int j = 0; j < n; j++)
				{
					if (!tbi(visited, j) || !tbi(minDegreeVisited, j)) continue;
					if (tbi(input, j) || Find(rep, j) != j || j == minDegreeNode || j == minNbNode) continue;
					commonNB++;
					currentDegree[indexByte(j)]--;
				}

				// Contract minimum degree node with minimum degree neighbour
				Union(rep, minNbNode, minDegreeNode);
				currentDegree[indexByte(Find(rep, minDegreeNode))] = (minDegree + minNbDegree - commonNB - 2);

				r--;
				mode = 0;
			}
		}

		for (int e = 0; e < repeat_outer && localQueueIn != localQueueOut; e++)
		{
			int curNode = localQueue[indexByte(localQueueIn)];
			int curNbCount = AdjList[curNode << maxDegreeLog];

			if (curNbIdx == curNbCount)
			{
				// We have finished exploring the current node
				curNbIdx = 0;
				localQueueIn++;
				if (localQueueIn >= localQueueSize)
					localQueueIn -= localQueueSize;
			}
			else
				for (int f = 0; f < repeat_inner && curNbIdx != curNbCount; f++)
				{
					curNbIdx++;
					int curNb = AdjList[(curNode << maxDegreeLog) + curNbIdx];

					// Check not already visited
					if (tbi(visited, curNb)) continue;

					// Mark visited
					sbi(visited, curNb);

					// Node in S (or not)?
					if (tbi(input, curNb) || (mode == 1 && Find(rep, curNb) == minDegreeNode) || (mode == 2 && Find(rep, curNb) == minNbNode))
					{
						// curNb is in S -> explore the path
						localQueue[indexByte(localQueueOut)] = curNb;
						localQueueOut++;
						if (localQueueOut >= localQueueSize)
							localQueueOut -= localQueueSize;
					}
					else
					{
						// (if mode 1) see if this is a min degree neighbour
						if (mode == 1 && currentDegree[indexByte(Find(rep, curNb))] < minNbDegree)
						{
							minNbNode = Find(rep, curNb);
							minNbDegree = currentDegree[indexByte(minNbNode)];
						}
						// (if mode 2) do nothing
					}
				}
		}
	}
#endif

	bool inSet;

	// Loop over nodes which are possible to eliminate at this point, add new solutions to output
	curExploreNode = 0;
	while (curExploreNode < n)
	{
		if(!tbi(elimCandidates, curExploreNode))
		{
			curExploreNode++;
			continue;
		}

		// Hash the current solution to pick a lock
		sbi(input, curExploreNode);
		ulong baseHash1 = Murmur3(input, 2169723734u); ulong baseHash2 = Murmur3(input, 2414063220u);
		cbi(input, curExploreNode);
		ulong lockHash = baseHash1;
		lockHash %= (1u << 30) - 35;
		lockHash %= (uint)(lockCount << 5);

		// Acquire one of the locks according to the hash table
		if (atomic_or(Locks + (int)(lockHash >> 5), 1u << (int)(lockHash & 31u)) & (1u << (int)(lockHash & 31u)))
			continue;

		inSet = true;

		// Bloom Filter: compute a number of hash functions, set bits - the previous lock is necessary to prevent multiple writes of the same set element at once (which might erroneously both return "not in set")
#pragma unroll
		for (int iter = 0; iter < hashFunctions; iter++)
		{
			ulong hashCode = baseHash1;
			baseHash1 += baseHash2;
			hashCode %= (1ul << 50) - 27;
			hashCode %= ((ulong)filterSize) << 5;

			// Atomically set the bit
			if (!(atomic_or(BloomFilter + (int)(hashCode >> 5), 1u << (int)(hashCode & 31)) & (1u << (int)(hashCode & 31))))
				inSet = false;
		}

		// Relase the lock
		atomic_and(Locks + (int)(lockHash >> 5), ~(1u << (int)(lockHash & 31u)));

		if (!inSet)
		{
			// Increment the number of generated states atomically to get an index into the output stack
			int myPos = atomic_inc(Out);

			if (myPos < nMax) {
				// Copy S from in to out
				for (int i = 0; i <= numBitsetWords; i++) OutStack[myPos * (numBitsetWords + 1) + i] = input[indexInt(i)];

				// Add curExploreNode to S
				sbi_norm(OutStack + myPos * (numBitsetWords + 1), curExploreNode);

				// Store last move
				OutStack[myPos * (numBitsetWords + 1) + numBitsetWords] <<= 8;
				OutStack[myPos * (numBitsetWords + 1) + numBitsetWords] |= curExploreNode;
			}
		}

		curExploreNode++;
	}
}

__kernel void ResetBloomFilter(volatile __global uint* BloomFilter, volatile __global int* Out, __global uint* InStack, int filterSize) {
	int idx = get_global_id(0);

	if(idx == 0) {
		Out[0] = 0;
		for(int i = 0; i <= numBitsetWords; i++) InStack[i] = 0;
	}

	idx *= resetSize;
	for(int i = 0; i < resetSize; i++) {
		if(idx >= filterSize) break;
		BloomFilter[idx] = 0;
		idx++;
	}
}