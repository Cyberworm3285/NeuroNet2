using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Mesh;
using NeuroNet2.Global;

namespace NeuroNet2.Neuro.Functions.Learning
{
    class BruteForce<T> : INeuroLearning<T>
    {
        public Func<T, T> Permutater { get; set; }
        public bool PrintProgress { get; set; }

        public BruteForce(Func<T, T> mutater)
        {
            Permutater = mutater;
        }

        public double LearnIteration(FullMesh<T> net, T[][] input, T[][] output, double switchThreshold)
        {
            double lastError = net.LastError;
            while (lastError > switchThreshold)
            {
                int x = GlobalRandom.Get.Next(0, net.Neurons.Count);
                int y = GlobalRandom.Get.Next(0, net.Neurons[x].Count);
                int w = GlobalRandom.Get.Next(0, net.Neurons[x][y].Weights.Count);
                net.Neurons[x][y].Weights[w] = Permutater(net.Neurons[x][y].Weights[w]);
                lastError = net.GetError(input.Select(i => net.Calc(i)).ToArray(), output);
                if (PrintProgress)
                    Console.WriteLine("Curr Error : " + lastError);
            }
            if (PrintProgress)
                Console.WriteLine("finished");
            return lastError;
        }
    }
}
