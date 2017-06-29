using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Cloo;

namespace TWGPU
{
    class Program
    {
        public const int numBitsetWords = 2; // Support 32x this number of vertices
        public const int maxDegreeLog = 7; // > base 2 log of (max degree plus 1)
        public const int localQueueSize = 48; // How many vertices can be in the local queue at once. Does not need to be larger than 32*numBitSetWords but could possibly be smaller (depending on the graph structure)

        public const int nMax = 1024 * 1024 * 180; // How much space (in items) to allocate for in/out-queue and bloom filter.

        public const int localWorkSize = 128; // 32 gives best performance for me, but seems on the low side...
        public const int resetSize = 256; // When resetting the bloom filter to all 0's, how large of a chunck of the array should be one work item
        
        public const int maxBatchSize = 60000; // Max number of simultaneous tasks in one kernel execution, using a too large number causes a time out
        public const int batchDelay = 2; // Milliseconds delay between two kernel executions. Setting this to something non-zero keeps your computer usable.

        public const int bloomFilterFactor = 24; // How many bits per element (nMax) to allocate in the bloom filter. Higher = lower false positive rate
        public const int hashFunctions = (int)(0.69314718056 * bloomFilterFactor + 0.5); // Approx bloomFilterFactor * ln2
        public const int lockCount = 65536; // How many locks to use to synchronize access to the filter

        public const bool UseMMW = false; // Whether MMW heuristic should be used

        public const bool UseLocalMemory = false; // Use local (shared) memory or private (global)?
        public const int repeat_inner = 10000, repeat_outer = 10000; // Loop unrolling: number of times the inner loop (outer loop) is executed before trying to see if an execution of the containing loop is needed.

        static DPAlgo algo;
        public static bool testCPU = true;

        static void Main(string[] args)
        {
            InitCL();

            HandleFile("../../../instances/selected/queen7_7.1.49.gr");

            Console.ReadLine();
        }

        static void HandleFile(string file)
        {
            if (!file.EndsWith(".gr") && !file.EndsWith(".dgf") && !file.EndsWith(".dimacs") && !file.EndsWith(".col")) return;

            algo = new DPAlgo();

            // Parse the input graph
            foreach (string line in File.ReadAllLines(file))
            {
                if (line.StartsWith("c") || line.StartsWith("n")) continue; // Comment or dimacs node

                if (line.StartsWith("p")) // Initialization
                {
                    string[] cf = line.Split(' ');
                    algo.n = int.Parse(cf[2]);
                    algo.m = int.Parse(cf[3]);
                    if (algo.n > 255 || algo.n > numBitsetWords * 32)
                    {
                        Console.WriteLine("Graph too large! (At most {0} nodes supported)", Math.Min(255, numBitsetWords * 32));
                        return;
                    }
                    
                    continue;
                }

                // Dimacs-style edge
                if (line.StartsWith("e"))
                {
                    string[] vt = line.Split(' '); // Edge

                    AddEdge(byte.Parse(vt[1]), byte.Parse(vt[2]));
                    continue;
                }

                // Something else, possibly an edge
                try
                {
                    string[] vt = line.Split(' '); // Edge
                    AddEdge(byte.Parse(vt[0]), byte.Parse(vt[1]));
                }
                catch { }
            }

            // Find an initial clique for the program to start with
            maxClique = new List<byte>();
            BronKerbosch(new List<byte>(), Enumerable.Range(0, algo.n).Select((x) => (byte)x).ToList(), new List<byte>());

            foreach (byte x in maxClique)
                DPAlgo.sbi(algo.Last, x);
            algo.LastCount = (byte)maxClique.Count;

            // Disjoint paths for the edge addition rule
            ComputeDisjointPaths();

            algo.Solve(true);

            if (testCPU)
            {
                algo.Solve(false);
            }
        }

        static void AddEdge(byte v1, byte v2)
        {
            if (v1 >= algo.n)
            {
                v1 = 0;
                Console.WriteLine("Node {0} exceeds n, assuming 0 was meant", v1);
            }

            if (v2 >= algo.n)
            {
                v2 = 0;
                Console.WriteLine("Node {0} exceeds n, assuming 0 was meant", v2);
            }

            DPAlgo.sbi(algo.AdjMatrix, (v1 << 8) + v2);
            DPAlgo.sbi(algo.AdjMatrix, (v2 << 8) + v1);

            algo.AdjList[(v1 << maxDegreeLog)]++;
            
            if (algo.AdjList[v1 << maxDegreeLog] >= (1 << maxDegreeLog) - 1)
                Console.WriteLine("Node {0} exceeds maximum degree", v1);
            else
                algo.AdjList[(v1 << maxDegreeLog) + algo.AdjList[v1 << maxDegreeLog]] = v2;

            algo.AdjList[(v2 << maxDegreeLog)]++;
            
            if (algo.AdjList[v2 << maxDegreeLog] >= (1 << maxDegreeLog) - 1)
                Console.WriteLine("Node {0} exceeds maximum degree", v2);
            else
                algo.AdjList[(v2 << maxDegreeLog) + algo.AdjList[v2 << maxDegreeLog]] = v1;
        }

        static List<byte> maxClique = new List<byte>();

        // Used to precompute a starting clique
        static void BronKerbosch(List<byte> R, List<byte> P, List<byte> X)
        {
            if (P.Count == 0 && X.Count == 0)
            {
                if (R.Count > maxClique.Count)
                    maxClique = R.ToList();
            }
            byte u = P.Union(X).FirstOrDefault();
            foreach (byte v in P.ToList())
            {
                if (DPAlgo.tbi(algo.AdjMatrix, (v << 8) + u)) continue;
                BronKerbosch(R.Union(new byte[] { v }).ToList(), P.Where((x) => DPAlgo.tbi(algo.AdjMatrix, (v << 8) + x)).ToList(), X.Where((x) => DPAlgo.tbi(algo.AdjMatrix, (v << 8) + x)).ToList());
                P.Remove(v);
                if (!X.Contains(v)) X.Add(v);
            }
        }

        // Precompute the number of vertex-disjoint paths between each pair of vertices
        static void ComputeDisjointPaths()
        {
            int[,] capacities = new int[2 * algo.n, 2 * algo.n];

            for (int x = 0; x < algo.n; x++)
            {
                capacities[2 * x, 2 * x + 1] = 1;
                for (int y = 0; y < algo.n; y++)
                    if (DPAlgo.tbi(algo.AdjMatrix, (x << 8) + y))
                        capacities[2 * x + 1, 2 * y] = 1;
            }
            
            for(int x = 0; x < algo.n; x++)
                for (int y = 0; y < algo.n; y++)
                {
                    if (DPAlgo.tbi(algo.AdjMatrix, (x << 8) + y))
                    {
                        algo.DisjointPaths[x, y] = 255;
                        continue;
                    }
                    if (x >= y) continue;

                    int s = 2 * x + 1;
                    int t = 2 * y;

                    int[,] currentFlow = new int[2 * algo.n, 2 * algo.n];
                    int F = 0;

                    while (true)
                    {
                        Queue<int> Q = new Queue<int>();
                        int[] par = Enumerable.Repeat(-1, 2 * algo.n).ToArray();
                        int[] m = new int[2 * algo.n];
                        par[s] = s;
                        m[s] = int.MaxValue;
                        Q.Enqueue(s);

                        while (Q.Count > 0)
                        {
                            int u = Q.Dequeue();
                            for (int v = 0; v < 2 * algo.n; v++)
                            {
                                if (par[v] != -1 || capacities[u, v] - currentFlow[u, v] <= 0) continue;
                                par[v] = u;
                                m[v] = Math.Min(m[u], capacities[u, v] - currentFlow[u, v]);
                                if (v == t)
                                {
                                    Q.Clear();
                                    break;
                                }
                                else
                                    Q.Enqueue(v);
                            }
                        }

                        if(m[t] == 0)
                            break;

                        int w = t;
                        while (w != s)
                        {
                            int u = par[w];
                            currentFlow[u, w] += m[t];
                            currentFlow[w, u] -= m[t];
                            w = u;
                        }

                        F += m[t];
                    }

                    algo.DisjointPaths[x, y] = (byte)F;
                    algo.DisjointPaths[y, x] = (byte)F;
                }
        }

        public static ComputeContext context;
        public static ComputeCommandQueue queue;
        public static ComputeProgram program;
        static void InitCL()
        {
            var platform = ComputePlatform.Platforms[0];
            Console.WriteLine("Initializing OpenCL... " + platform.Name + " (" + platform.Profile + ").");
            Console.WriteLine(platform.Version);
            Console.WriteLine(platform.Devices.First().Name + " (" + platform.Devices.First().Type + ")");
            Console.WriteLine("SMs: {0}, Max Work Group Size: {1}", platform.Devices.First().MaxComputeUnits, platform.Devices.First().MaxWorkGroupSize);
            Console.WriteLine((platform.Devices.First().GlobalMemorySize / 1024 / 1024) + " MiB global memory / " + (platform.Devices.First().LocalMemorySize / 1024) + " KiB local memory");
            Console.WriteLine("Local memory type: " + platform.Devices.First().LocalMemoryType);

            long globalMem = 32L * nMax * (numBitsetWords * 2 + 2); // instack, outstack
            globalMem += (long)nMax * bloomFilterFactor; // bloom filter
            globalMem += lockCount * 32L; // lock array
            globalMem += 8 * 256 << maxDegreeLog; // adj list

            int localMem = localQueueSize * 8;
            localMem += (numBitsetWords * 3 + 2) * 32;
            if (UseMMW)
            {
                localMem += (numBitsetWords * 2 * 32) * 8 + (numBitsetWords * 1) * 32;
            }
            localMem *= localWorkSize;

            Console.WriteLine("Current settings use estimated " + (globalMem / 8 / 1024 / 1024) + " MiB global memory / " + (localMem / 8 / 1024) + " KiB local memory");

            double falsePosRate = Math.Pow(0.5, Math.Log(2) * bloomFilterFactor);
            Console.WriteLine("Bloom Filter load factor {0}, {1} hash functions, false positive rate 1 in {2}", bloomFilterFactor, hashFunctions, 1000 * ((int)(1.0 / falsePosRate) / 1000));
            
            context = new ComputeContext(ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
            //context = new ComputeContext(ComputeDeviceTypes.Cpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
            
            var streamReader = new StreamReader("../../../treewidth.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();
            
            clSource = "#define nMax " + nMax + " \r\n" + clSource;
            clSource = "#define localQueueSize " + localQueueSize + " \r\n" + clSource;
            clSource = "#define hashFunctions " + hashFunctions + " \r\n" + clSource;
            clSource = "#define localWorkSize " + localWorkSize + " \r\n" + clSource;
            clSource = "#define lockCount " + lockCount + " \r\n" + clSource;
            clSource = "#define maxDegreeLog " + maxDegreeLog + " \r\n" + clSource;
            clSource = "#define numBitsetWords " + numBitsetWords + " \r\n" + clSource;
            clSource = "#define resetSize " + resetSize + " \r\n" + clSource;
            clSource = "#define repeat_inner " + repeat_inner + " \r\n" + clSource;
            clSource = "#define repeat_outer " + repeat_outer + " \r\n" + clSource;
            if (Program.UseMMW) clSource = "#define useMMW \r\n" + clSource;
            if (Program.UseLocalMemory) clSource = "#define useLocalMemory \r\n" + clSource;

            // Try compiling code
            program = new ComputeProgram(context, clSource);
            try
            {
                //-cl-strict-aliasing  "-cl-nv-verbose"
                program.Build(null, null, null, IntPtr.Zero);
                Console.WriteLine(program.GetBuildLog(context.Devices[0]));
                File.WriteAllBytes("kernel.bin", program.Binaries.First());
            }
            catch
            {
                Console.WriteLine("Error building OpenCL code");
                Console.WriteLine(program.GetBuildLog(context.Devices[0]));
                File.WriteAllBytes("kernel.bin", program.Binaries.First());
                Console.ReadLine();
            }
            
            queue = new ComputeCommandQueue(context, context.Devices[0], ComputeCommandQueueFlags.None);
        }
    }
}
