using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Cloo;

namespace TWGPU
{
    class DPAlgo
    {
        public uint[] AdjMatrix = new uint[256 * 256 / 32];
        public byte[] AdjList = new byte[(256 << Program.maxDegreeLog)];
        public uint[] Last = new uint[Program.numBitsetWords];
        public byte[,] DisjointPaths = new byte[256, 256];
        public byte LastCount = 0;
        public int n, m;

        int k;
        uint[] InStack = new uint[Program.nMax * (Program.numBitsetWords + 1)]; // bits for the bistring, + 1 for the history
        uint[] OutStack = new uint[Program.nMax * (Program.numBitsetWords + 1)]; // bits for the bistring, + 1 for the history
        uint[] BloomFilter = new uint[((long)Program.nMax * Program.bloomFilterFactor) / 32]; // Bloom Filter
        uint[] Locks = new uint[Program.lockCount]; // Used to synchronize bloom filter access

        // How many ints are in input/output
        int nIn, nOut;

        int[] OutArray = new int[6];

        ComputeKernel twKernel, resetKernel;
        ComputeBuffer<int> OutBuffer;
        ComputeBuffer<uint> InStackBuffer;
        ComputeBuffer<uint> OutStackBuffer;
        ComputeBuffer<uint> BloomFilterBuffer;
        ComputeBuffer<uint> LockBuffer;
        ComputeBuffer<uint> LastBuffer;
        ComputeBuffer<byte> AdjListBuffer;

        byte[] AdjListNew;
        public DPAlgo()
        {
            twKernel = Program.program.CreateKernel("DPAlgoIteration");
            resetKernel = Program.program.CreateKernel("ResetBloomFilter");

            var flags_rw = ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer;
            var flags_w = ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer;
            var flags_r = ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer;

            OutBuffer = new ComputeBuffer<int>(Program.context, flags_rw, OutArray);
            InStackBuffer = new ComputeBuffer<uint>(Program.context, ComputeMemoryFlags.ReadWrite, InStack.Length);
            OutStackBuffer = new ComputeBuffer<uint>(Program.context, ComputeMemoryFlags.ReadWrite, OutStack.Length);
            BloomFilterBuffer = new ComputeBuffer<uint>(Program.context, ComputeMemoryFlags.ReadWrite, BloomFilter.Length);
            LockBuffer = new ComputeBuffer<uint>(Program.context, flags_rw, Locks);
            LastBuffer = new ComputeBuffer<uint>(Program.context, flags_rw, Last);
            AdjListBuffer = new ComputeBuffer<byte>(Program.context, flags_r, AdjList);

            twKernel.SetMemoryArgument(3, OutBuffer);
            twKernel.SetMemoryArgument(4, InStackBuffer);
            twKernel.SetMemoryArgument(5, OutStackBuffer);
            twKernel.SetMemoryArgument(6, BloomFilterBuffer);
            twKernel.SetMemoryArgument(7, LockBuffer);
            twKernel.SetMemoryArgument(8, LastBuffer);
            twKernel.SetMemoryArgument(9, AdjListBuffer);

            resetKernel.SetMemoryArgument(0, BloomFilterBuffer);
            resetKernel.SetMemoryArgument(1, OutBuffer);
        }

        public void Solve(bool GPU)
        {
            Console.WriteLine("Starting Solve ({0} vertices, {1} edges) {2}", n, m, GPU ? "GPU" : "CPU");
            long totalExp = 0;
            Stopwatch sw = new Stopwatch(), swKernelTotal = new Stopwatch(), swKernel = new Stopwatch();
            sw.Start();

            int kExceednMax = int.MaxValue;

            if (GPU)
            {
                twKernel.SetValueArgument(0, n);
                Program.queue.WriteToBuffer(Last, LastBuffer, false, null);
                Program.queue.WriteToBuffer(AdjList, AdjListBuffer, false, null);
            }

            // Try increasing treewidth values
            for (k = Math.Max(LastCount - 1, 0); k < n; k++)
            {
                nOut = 1;
                for (int i = 0; i <= Program.numBitsetWords; i++) OutStack[i] = 0;

                // Edge addition rule
                AdjListNew = new byte[AdjList.Length];
                int newEdge = 0;
                for (int i = 0; i < AdjListNew.Length; i++)
                    AdjListNew[i] = AdjList[i];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        if (DisjointPaths[i, j] >= k + 2 && !tbi(AdjMatrix, (i << 8) + j))
                        {
                            newEdge++;
                            AdjListNew[(i << Program.maxDegreeLog)]++;
                            AdjListNew[(i << Program.maxDegreeLog) + AdjListNew[i << Program.maxDegreeLog]] = (byte)j;
                        }

                if (GPU)
                    Program.queue.WriteToBuffer(AdjListNew, AdjListBuffer, false, null);

                // Loop over size of S
                for (int z = 0; z < n - Math.Max(LastCount, k + 1); z++)
                {
                    if (kExceednMax < k) break;

                    if (nOut >= Program.nMax)
                        nOut = Program.nMax - 1;

                    nIn = nOut;
                    nOut = 0;
                    totalExp += nIn;

                    if (nIn == 0)
                        break;

                    int newFilterSize = (int)(Math.Min((long)nIn * n * Program.bloomFilterFactor, (long)Program.nMax * Program.bloomFilterFactor) / 32);

                    if (GPU)
                    {
                        if (z == 0 || z % 2 == 1)
                            resetKernel.SetMemoryArgument(2, InStackBuffer);
                        else
                            resetKernel.SetMemoryArgument(2, OutStackBuffer);
                        resetKernel.SetValueArgument(3, newFilterSize);
                        Program.queue.Execute(resetKernel, null, new long[] { (newFilterSize + Program.resetSize - 1) / Program.resetSize }, null, null);
                        Program.queue.Finish();

                        twKernel.SetMemoryArgument(4 + (z % 2), InStackBuffer);
                        twKernel.SetMemoryArgument(5 - (z % 2), OutStackBuffer);

                        twKernel.SetValueArgument(1, k);
                        twKernel.SetValueArgument(10, newFilterSize);

                        twKernel.SetValueArgument(12, z);

                        int numBatches = ((nIn + 9) / 10 + Program.maxBatchSize - 1) / Program.maxBatchSize;

                        swKernel.Reset();
                        for (int i = 0; i < numBatches; i++)
                        {
                            int start = (int)(((long)nIn * i) / numBatches);
                            int end = (int)(((long)nIn * (i + 1)) / numBatches);
                            int num = end - start;
                            twKernel.SetValueArgument(11, start);
                            twKernel.SetValueArgument(2, nIn);
                            swKernel.Start(); swKernelTotal.Start();
                            Program.queue.Execute(twKernel, null, new long[] { Program.localWorkSize * ((num + Program.localWorkSize - 1) / Program.localWorkSize) }, new long[] { Program.localWorkSize }, null);
                            Program.queue.Finish();
                            swKernel.Stop(); swKernelTotal.Stop(); sw.Stop();
                            if (Program.batchDelay > 0 && i + 1 < numBatches) System.Threading.Thread.Sleep(Program.batchDelay);
                            sw.Start();
                        }

                        Program.queue.ReadFromBuffer(OutBuffer, ref OutArray, true, null);
                        nOut = (int)OutArray[0];
                    }
                    else
                    {
                        uint[] temp = InStack;
                        InStack = OutStack;
                        OutStack = temp;

                        BloomFilter = new uint[newFilterSize];

                        uint[] visited = new uint[Program.numBitsetWords];
                        uint[] elimCandidates = new uint[Program.numBitsetWords];
                        byte[] localQueue = new byte[Program.localQueueSize];

                        // For computing MMW
                        byte[] currentDegree = new byte[n]; // Degree of node
                        byte[] rep = new byte[n]; // Union-find
                        uint[] minDegreeVisited = new uint[Program.numBitsetWords];

                        swKernelTotal.Start(); swKernel.Restart();
                        for (int i = 0; i < nIn; i++)
                        {
                            int curExploreNode = -1;
                            int curCount = 255;
                            int curNbIdx = 0;
                            int localQueueIn = 0, localQueueOut = 0;

                            for (int j = 0; j < Program.numBitsetWords; j++)
                                elimCandidates[j] = 0;

                            if (Program.UseMMW)
                            {
                                for (int j = 0; j < n; j++)
                                {
                                    currentDegree[j] = 0; rep[j] = 0;
                                }
                                for (int j = 0; j < Program.numBitsetWords; j++)
                                    minDegreeVisited[j] = 0;
                            }

                            // Compute which nodes can be feasibly eliminated
                            while (true)
                            {
                                // Start exploring a new node
                                if (localQueueIn == localQueueOut)
                                {
                                    // curExploreNode can be eliminated at this point, but only if it's not prohibited due to it being in last
                                    if (curCount <= k && (!Program.UseMMW || !tbi(Last, curExploreNode)))
                                        sbi(elimCandidates, curExploreNode);

                                    // Remember the degree for use in computing MMW
                                    if (Program.UseMMW && curExploreNode < n && curExploreNode != -1)
                                        currentDegree[curExploreNode] = (byte)curCount;

                                    // Let's check the next node
                                    curExploreNode++;

                                    // We finished exploring all the nodes
                                    if (curExploreNode >= n) break;

                                    // For MMW: initialize reps
                                    if (Program.UseMMW)
                                        rep[curExploreNode] = (byte)curExploreNode;

                                    // Check that node is not already in S, if not, put it on the queue.
                                    if (!tbi(InStack, i * (Program.numBitsetWords + 1), curExploreNode) && (Program.UseMMW || !tbi(Last, curExploreNode)))
                                    {
                                        localQueueIn = 0; localQueueOut = 1;
                                        localQueue[0] = (byte)curExploreNode;
                                        curCount = 0;
                                        curNbIdx = 0;
                                        for (int j = 0; j < Program.numBitsetWords; j++) visited[j] = 0;
                                        sbi(visited, curExploreNode);
                                    }
                                    else
                                    {
                                        curCount = 255;
                                        if (Program.UseMMW)
                                            currentDegree[curExploreNode] = 255; // Set the degree of nodes in S to some very high amount
                                    }
                                }

                                // The queue is (now) non-empty
                                for (int e = 0; e < Program.repeat_outer && localQueueIn != localQueueOut; e++)
                                {
                                    int curNode = localQueue[localQueueIn];
                                    int curNbCount = AdjList[curNode << Program.maxDegreeLog];

                                    if (curNbIdx == curNbCount)
                                    {
                                        // We have finished exploring the current node
                                        curNbIdx = 0;
                                        localQueueIn++;
                                        if (localQueueIn >= Program.localQueueSize)
                                            localQueueIn -= Program.localQueueSize;
                                    }
                                    else
                                        for (int f = 0; f < Program.repeat_inner && curNbIdx != curNbCount; f++)
                                        {
                                            curNbIdx++;
                                            int curNb = AdjList[(curNode << Program.maxDegreeLog) + curNbIdx];

                                            // Check not already visited
                                            if (tbi(visited, curNb)) continue;

                                            // Mark visited
                                            sbi(visited, curNb);

                                            if (tbi(InStack, i * (Program.numBitsetWords + 1), curNb))
                                            {
                                                // curNb is in S -> explore the path
                                                localQueue[localQueueOut++] = (byte)curNb;
                                                if (localQueueOut >= Program.localQueueSize)
                                                    localQueueOut -= Program.localQueueSize;

                                                if (localQueueOut == localQueueIn)
                                                    throw new Exception("queue overflow");
                                            }
                                            else
                                            {
                                                // curNb is not in S -> count towards the count
                                                curCount++;
                                                // If curCount becomes too high, terminate BFS (not possible with MMW, need to know exact degree)
                                                if (!Program.UseMMW && curCount > k)
                                                {
                                                    localQueueIn = localQueueOut;
                                                    break;
                                                }
                                            }
                                        }
                                }
                            }


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
                            while (r > k - 1 && Program.UseMMW)
                            {
                                if (localQueueIn == localQueueOut)
                                {
                                    // a BFS has terminated, do something
                                    if (mode == 0)
                                    {
                                        // Find a minimum degree node, and start BFS to find minimum degree neighbour
                                        minDegree = 255; secondMinDegree = 255;
                                        minDegreeNode = 255;
                                        for (int j = 0; j < n; j++)
                                            if (currentDegree[Find(rep, j)] < minDegree)
                                            {
                                                secondMinDegree = minDegree;
                                                minDegreeNode = Find(rep, j);
                                                minDegree = currentDegree[minDegreeNode];
                                            }
                                            else if (currentDegree[Find(rep, j)] == minDegree)
                                                secondMinDegree = minDegree;

                                        if (secondMinDegree == 255)
                                            secondMinDegree = minDegree;

                                        if (secondMinDegree > k)
                                            break; // return in the GPU version

                                        // Do a BFS to find the neighbour with minimum degree
                                        minNbDegree = 255;
                                        minNbNode = 255;

                                        localQueueIn = 0; localQueueOut = 1;
                                        localQueue[0] = (byte)minDegreeNode;
                                        curNbIdx = 0;
                                        for (int j = 0; j < Program.numBitsetWords; j++) visited[j] = 0;
                                        sbi(visited, minDegreeNode);

                                        mode = 1;
                                    }
                                    else if (mode == 1)
                                    {
                                        // We found the minimum degree neighbour, now start BFS to compute its adjacency list
                                        for (int j = 0; j < Program.numBitsetWords; j++) minDegreeVisited[j] = visited[j];

                                        localQueueIn = 0; localQueueOut = 1;
                                        localQueue[0] = (byte)minNbNode;
                                        curNbIdx = 0;
                                        for (int j = 0; j < Program.numBitsetWords; j++) visited[j] = 0;
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

                                        int commonNB = 0; int ca = 0, cb = 0;

                                        for (int j = 0; j < n; j++)
                                        {
                                            if (!tbi(InStack, i * (Program.numBitsetWords + 1), j) && tbi(visited, j) && Find(rep, j) == j) ca++;
                                            if (!tbi(InStack, i * (Program.numBitsetWords + 1), j) && tbi(minDegreeVisited, j) && Find(rep, j) == j) cb++;

                                            if (!tbi(visited, j) || !tbi(minDegreeVisited, j)) continue;
                                            if (tbi(InStack, i * (Program.numBitsetWords + 1), j) || Find(rep, j) != j || j == minDegreeNode || j == minNbNode) continue;
                                            commonNB++;
                                            currentDegree[j]--;
                                        }

                                        // Contract minimum degree node with minimum degree neighbour
                                        Union(rep, minNbNode, minDegreeNode);
                                        currentDegree[Find(rep, minDegreeNode)] = (byte)(minDegree + minNbDegree - commonNB - 2);

                                        r--;
                                        mode = 0;
                                    }
                                }

                                for (int e = 0; e < Program.repeat_outer && localQueueIn != localQueueOut; e++)
                                {
                                    int curNode = localQueue[localQueueIn];
                                    int curNbCount = AdjList[curNode << Program.maxDegreeLog];

                                    if (curNbIdx == curNbCount)
                                    {
                                        // We have finished exploring the current node
                                        curNbIdx = 0;
                                        localQueueIn++;
                                        if (localQueueIn > Program.localQueueSize)
                                            localQueueIn -= Program.localQueueSize;
                                    }
                                    else
                                        for (int f = 0; f < Program.repeat_inner && curNbIdx != curNbCount; f++)
                                        {
                                            curNbIdx++;
                                            int curNb = AdjList[(curNode << Program.maxDegreeLog) + curNbIdx];

                                            // Check not already visited
                                            if (tbi(visited, curNb)) continue;

                                            // Mark visited
                                            sbi(visited, curNb);

                                            // Node in S (or not)?
                                            if (tbi(InStack, i * (Program.numBitsetWords + 1), curNb) || (mode == 1 && Find(rep, curNb) == minDegreeNode) || (mode == 2 && Find(rep, curNb) == minNbNode))
                                            {
                                                // curNb is in S -> explore the path
                                                localQueue[localQueueOut++] = (byte)curNb;
                                                if (localQueueOut > Program.localQueueSize)
                                                    localQueueOut -= Program.localQueueSize;

                                                if (localQueueOut == localQueueIn)
                                                    throw new Exception("queue overflow");
                                            }
                                            else
                                            {
                                                // (if mode 1) see if this is a min degree neighbour
                                                if (mode == 1 && currentDegree[Find(rep, curNb)] < minNbDegree)
                                                {
                                                    minNbNode = Find(rep, curNb);
                                                    minNbDegree = currentDegree[minNbNode];
                                                }
                                                // (if mode 2) do nothing
                                            }
                                        }
                                }
                            }

                            // Not necessary in GPU version, since this outer loop simply does not exist there
                            if (secondMinDegree > k)
                                continue;

                            // Loop over nodes which are possible to eliminate at this point, add new solutions to output
                            curExploreNode = 0;
                            while (curExploreNode < n)
                            {
                                if (!tbi(elimCandidates, curExploreNode))
                                {
                                    curExploreNode++;
                                    continue;
                                }

                                // Hash the current solution to pick a lock
                                sbi(InStack, i * (Program.numBitsetWords + 1), curExploreNode);
                                ulong baseHash1 = Murmur3(InStack, i * (Program.numBitsetWords + 1), 2169723734u); ulong baseHash2 = Murmur3(InStack, i * (Program.numBitsetWords + 1), 2414063220u);
                                cbi(InStack, i * (Program.numBitsetWords + 1), curExploreNode);
                                ulong lockHash = baseHash1;
                                lockHash %= (1u << 30) - 35;
                                lockHash %= (uint)(Program.lockCount << 5);

                                // Convert to atomic operation on GPU
                                if (tsbi(Locks, (uint)lockHash))
                                    continue;

                                bool In_Set = true;

                                // Bloom Filter: compute 10 hash functions, set bits
                                for (int iter = 0; iter < Program.hashFunctions; iter++)
                                {
                                    ulong hashCode = baseHash1;
                                    baseHash1 += baseHash2;
                                    hashCode %= (1ul << 50) - 27;
                                    hashCode %= (((ulong)newFilterSize) << 5);

                                    // Convert to atomic operation on GPU
                                    if (!tsbi(BloomFilter, hashCode))
                                        In_Set = false;
                                }

                                if (!In_Set)
                                {
                                    // Convert to atomic operation on GPU
                                    int myPos = nOut++;

                                    if (myPos < Program.nMax)
                                    {
                                        // Copy S from in to out
                                        for (int j = 0; j <= Program.numBitsetWords; j++)
                                            OutStack[myPos * (Program.numBitsetWords + 1) + j] = InStack[i * (Program.numBitsetWords + 1) + j];

                                        // Add curExploreNode to S
                                        sbi(OutStack, myPos * (Program.numBitsetWords + 1), curExploreNode);

                                        // Store last move
                                        OutStack[myPos * (Program.numBitsetWords + 1) + Program.numBitsetWords] <<= 8;
                                        OutStack[myPos * (Program.numBitsetWords + 1) + Program.numBitsetWords] |= (byte)curExploreNode;
                                    }
                                }

                                curExploreNode++;

                                // Convert to atomic operation on GPU
                                cbi(Locks, (int)lockHash);
                            }
                        }

                        swKernelTotal.Stop(); swKernel.Stop();
                    }

                    Console.Title = String.Format("k={0} |S|={1} n={2} {4}/s ({3}%)", k, nIn, z, Math.Round(100.0 * nOut / Program.nMax, 1), Math.Round(nIn / (swKernel.ElapsedMilliseconds / 1000.0), 0));

                    if (nOut >= Program.nMax)
                    {
                        Console.WriteLine("Warning! nMax exceeded (k={0}).", k);
                        kExceednMax = Math.Min(kExceednMax, k);
                    }
                }

                // Found solution
                if (nOut != 0)
                    break;
            }

            if (GPU)
            {
                Program.queue.Finish();

                // Dispose OpenCL resources
                OutBuffer.Dispose();
                InStackBuffer.Dispose();
                OutStackBuffer.Dispose();
                BloomFilterBuffer.Dispose();
                LockBuffer.Dispose();
                LastBuffer.Dispose();
                AdjListBuffer.Dispose();
                twKernel.Dispose();
                resetKernel.Dispose();

                Program.queue.Finish();
            }

            sw.Stop();
            Console.WriteLine("Solve Done, treewidth={0}, time={1}ms, kernel time={2}ms, exp={3}", k, sw.ElapsedMilliseconds.ToString(), swKernelTotal.ElapsedMilliseconds, totalExp);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static uint getWordHash(uint[] str, int offset, uint multiplier)
        {
            uint hash = 0;
            for (int i = Program.numBitsetWords - 1; i >= 0; i--)
            {
                hash += str[i + offset] * multiplier;
                int shifted = (int)(multiplier << 5);
                multiplier = (uint)(shifted - multiplier);
            }
            return hash;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sbi(uint[] arr, ulong pos)
        {
            arr[pos >> 5] |= (1u << (int)(pos & 31));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sbi(uint[] arr, uint pos)
        {
            arr[pos >> 5] |= (1u << (int)(pos & 31));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sbi(uint[] arr, int pos)
        {
            arr[pos >> 5] |= (1u << (pos & 31));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void cbi(uint[] arr, int pos)
        {
            arr[pos >> 5] &= ~(1u << (pos & 31));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool tbi(uint[] arr, ulong pos)
        {
            return (arr[pos >> 5] & (1u << (int)(pos & 31))) != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool tbi(uint[] arr, uint pos)
        {
            return (arr[pos >> 5] & (1u << (int)(pos & 31))) != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sbi(uint[] arr, int offset, uint pos)
        {
            arr[offset + (pos >> 5)] |= (1u << (int)(pos & 31));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool tbi(uint[] arr, int pos)
        {
            return (arr[pos >> 5] & (1u << (pos & 31))) != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sbi(uint[] arr, int offset, int pos)
        {
            arr[offset + (pos >> 5)] |= (1u << (pos & 31));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void cbi(uint[] arr, int offset, int pos)
        {
            arr[offset + (pos >> 5)] &= ~(1u << (pos & 31));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool tbi(uint[] arr, int offset, int pos)
        {
            return (arr[offset + (pos >> 5)] & (1u << (pos & 31))) != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool tsbi(uint[] arr, int pos)
        {
            bool ret = tbi(arr, pos);
            sbi(arr, pos);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool tsbi(uint[] arr, ulong pos)
        {
            bool ret = tbi(arr, pos);
            sbi(arr, pos);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool tsbi(uint[] arr, uint pos)
        {
            bool ret = tbi(arr, pos);
            sbi(arr, pos);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Find(byte[] rep, int x)
        {
            int y = x;
            while (y != rep[y])
                y = rep[y];

            while (x != rep[x])
            {
                int z = rep[x];
                rep[x] = (byte)y;
                x = z;
            }

            return y;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Union(byte[] rep, int x, int y)
        {
            rep[Find(rep, x)] = (byte)Find(rep, y);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Murmur3(uint[] key, int offset, uint seed)
        {
            uint h = seed;


            for (int i = 0; i < Program.numBitsetWords; i++)
            {
                uint k = key[offset + i];
                k *= 0xcc9e2d51;
                k = (k << 15) | (k >> 17);
                k *= 0x1b873593;
                h ^= k;
                h = (h << 13) | (h >> 19);
                h += (h << 2) + 0xe6546b64;
            }

            h ^= Program.numBitsetWords << 2;
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;

            return h;

        }
    }
}
