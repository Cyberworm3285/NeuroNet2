using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Mesh;
using NeuroNet2.Global;

namespace NeuroNet2.Neuro.Functions.Learning
{
    class Evolution<T> : INeuroLearning<T>
    {
        public Func<T, T> Permutater { get; set; }
        public bool PrintProgress { get; set; }

        public Evolution(Func<T, T> mutater)
        {
            Permutater = mutater;
        }

    public double LearnIteration(FullMesh<T> net, T[][] input, T[][] output, double switchThreshold)
        {
            double temp;
            int x = GlobalRandom.Get.Next(0, net.Neurons.Count);
            int y = GlobalRandom.Get.Next(0, net.Neurons[x].Count);
            int w = GlobalRandom.Get.Next(0, net.Neurons[x][y].Weights.Count);
            T old = net.Neurons[x][y].Weights[w];
            net.Neurons[x][y].Weights[w] = Permutater(net.Neurons[x][y].Weights[w]);
            temp = net.GetError(input.Select(i => net.Calc(i)).ToArray(), output);

            if (temp - net.LastError < switchThreshold)
            {
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
