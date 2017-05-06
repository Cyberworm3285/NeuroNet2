using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Mesh;
using NeuroNet2.Global;

namespace NeuroNet2.Neuro.Functions.Learning
{
    class BruteForce<TInput, TOutput, TWeight, TActivator> : INeuroLearning<TInput, TOutput, TWeight, TActivator>
    {
        public Func<TWeight, TWeight> Permutater { get; set; }
        public Func<Func<TInput, TWeight, TInput>, Func<TInput, TWeight, TInput>> WeightingChanging { get; set; }
        public Func<Func<TActivator, TOutput>, Func<TActivator, TOutput>> ActivationChanging { get; set; }
        public bool PrintProgress { get; set; }

        public BruteForce(Func<TWeight, TWeight> mutater, Func<Func<TInput, TWeight, TInput>, Func<TInput, TWeight, TInput>> weightingChange, Func<Func<TActivator, TOutput>, Func<TActivator, TOutput>> activationChange)
        {
            Permutater = mutater;
            WeightingChanging = weightingChange;
            ActivationChanging = activationChange;
        }

        public double LearnIteration(FullMesh<TInput, TOutput, TWeight, TActivator> net, TInput[][] input, TOutput[][] output, double switchThreshold)
        {
            double lastError = net.LastError;
            int counter = 1;
            while (lastError > switchThreshold)
            {
                int x = GlobalRandom.Get.Next(0, net.Neurons.Count);
                int y = GlobalRandom.Get.Next(0, net.Neurons[x].Count);
                int w = GlobalRandom.Get.Next(0, net.Neurons[x][y].Weights.Count);
                net.Neurons[x][y].Weights[w] = Permutater(net.Neurons[x][y].Weights[w]);
                net.Neurons[x][y].WeightingFunction = WeightingChanging(net.Neurons[x][y].WeightingFunction);
                net.Neurons[x][y].ActivationFunction = ActivationChanging(net.Neurons[x][y].ActivationFunction);
                lastError = net.GetError(input.Select(i => net.Calc(i)).ToArray(), output);
                if (PrintProgress)
                    Console.WriteLine(counter ++ + ": Curr Error : " + lastError);
            }
            if (PrintProgress)
                Console.WriteLine("finished");
            return lastError;
        }
    }
}
