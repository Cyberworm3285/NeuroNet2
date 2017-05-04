using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroNet2.Neuro.Mesh
{
    class Neuron<T>
    {
        public List<T> Weights { get; set; }

        public Func<T, T> ActivationFunction { get; set; }
        public Func<T, T, T> AdderFunction { get; set; }
        public Func<T, T, T> WeightingFunction { get; set; }

        public int LayerIndex { get; }
        public FullMesh<T> OwnerNet { get; }

        public Neuron(Func<T,T> activation, Func<T,T,T> adder, Func<T,T,T> weighting, int layer, FullMesh<T> owner)
        {
            ActivationFunction = activation;
            AdderFunction = adder;
            WeightingFunction = weighting;

            LayerIndex = layer;
            OwnerNet = owner;
        }

        public T Calc()
        {
            if (LayerIndex == 0)
            {
                T temp = WeightingFunction(OwnerNet.Input.First(), Weights.First());
                for (int i = 1; i < OwnerNet.Input.Length; i++)
                    temp = AdderFunction(temp, WeightingFunction(OwnerNet.Input[i], Weights[i]));
                return ActivationFunction(temp);
            }
            else
            {
                T temp = WeightingFunction(OwnerNet.Neurons[LayerIndex - 1].First().Calc(), Weights.First());
                for (int i = 1; i < Weights.Count; i++)
                    temp = AdderFunction(temp, WeightingFunction(OwnerNet.Neurons[LayerIndex - 1][i].Calc(), Weights[i]));
                return ActivationFunction(temp);
            }
        }
    }
}
