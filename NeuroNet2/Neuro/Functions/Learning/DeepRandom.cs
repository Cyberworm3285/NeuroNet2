using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuroNet2.Neuro.Mesh;

using NeuroNet2.Global;

namespace NeuroNet2.Neuro.Functions.Learning
{
    class DeepRandom<TInput, TOutput, TWeight, TActivator> : INeuroLearning<TInput, TOutput, TWeight, TActivator>
    {
        public int InnerIterations { get; private set; }
        public Func<TWeight, TWeight> Permutater { get; set; }
        public bool PrintProgress { get; set; }

        public DeepRandom(int iterations, Func<TWeight,TWeight> mutater)
        {
            InnerIterations = iterations;
            Permutater = mutater;
        }

        public double LearnIteration(FullMesh<TInput, TOutput, TWeight, TActivator> net, TInput[][] input, TOutput[][] output, double switchThreshold = -0.1)
        {
            double lastError = net.LastError;
            double temp = 0;
            foreach (List<Neuron<TInput, TOutput, TWeight, TActivator>> l in net.Neurons)
            {
                if (PrintProgress)
                    Console.WriteLine("new Layer");
                for (int i = 0; i < InnerIterations; i++)
                {
                    int y = GlobalRandom.Get.Next(0, l.Count);
                    int w = GlobalRandom.Get.Next(0, l[y].Weights.Count);
                    TWeight old = l[y].Weights[w];
                    l[y].Weights[w] = Permutater(l[y].Weights[w]);
                    TOutput[][] c = input.Select(t => net.Calc(t)).ToArray();
                    temp = net.GetError(c, output);
                    if (!(temp - lastError < switchThreshold))
                    {
                        l[y].Weights[w] = old;
                    }
                    else
                    {
                        lastError = temp;
                        if (PrintProgress)
                            Console.WriteLine("new Error : " + temp);
                    }
                }
            }
            if (PrintProgress)
                Console.WriteLine("finished");
            return lastError;
        }
    }
}
