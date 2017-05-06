using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Mesh;
using NeuroNet2.Global;

namespace NeuroNet2.Neuro.Functions.Learning
{
    class Evolution<TInput, TOutput, TWeight, TActivator> : INeuroLearning<TInput, TOutput, TWeight, TActivator>
    {
        public Func<TWeight, TWeight> Permutater { get; set; }
        public bool PrintProgress { get; set; }

        public Evolution(Func<TWeight, TWeight> mutater)
        {
            Permutater = mutater;
        }

    public double LearnIteration(FullMesh<TInput, TOutput, TWeight, TActivator> net, TInput[][] input, TOutput[][] output, double switchThreshold)
        {
            double temp;
            int x = GlobalRandom.Get.Next(0, net.Neurons.Count);
            int y = GlobalRandom.Get.Next(0, net.Neurons[x].Count);
            int w = GlobalRandom.Get.Next(0, net.Neurons[x][y].Weights.Count);
            TWeight old = net.Neurons[x][y].Weights[w];
            net.Neurons[x][y].Weights[w] = Permutater(net.Neurons[x][y].Weights[w]);
            temp = net.GetError(input.Select(i => net.Calc(i)).ToArray(), output);

            if (temp - net.LastError < switchThreshold)
            {
                Console.WriteLine("new Error : " + temp);
                return temp;
            }
            else
            {
                net.Neurons[x][y].Weights[w] = old;
                return net.LastError;
            }
        }
    }
}
