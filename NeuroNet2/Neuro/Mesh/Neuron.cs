using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroNet2.Neuro.Mesh
{
    class Neuron<TInput, TOutput, TWeight, TActivator>
    {
        public List<TWeight> Weights { get; set; }

        public Converter<TOutput, TInput> InterchangeConverter { get; set; }
        public Func<TActivator, TOutput> ActivationFunction { get; set; }
        public Converter<TInput, TActivator> InitialConverter { get; set; }
        public Func<TActivator, TInput, TActivator> AdderFunction { get; set; }
        public Func<TInput, TWeight, TInput> WeightingFunction { get; set; }

        public int LayerIndex { get; }
        public FullMesh<TInput, TOutput, TWeight, TActivator> OwnerNet { get; }

        public Neuron(
            Converter<TOutput, TInput> interchangeCon,
            Func<TActivator, TOutput> activation, 
            Converter<TInput, TActivator> initCon,
            Func<TActivator, TInput, TActivator> adder, 
            Func<TInput, TWeight, TInput> weighting,
            int layer, 
            FullMesh<TInput, TOutput, TWeight, TActivator> owner)
        {
            InterchangeConverter = interchangeCon;
            ActivationFunction = activation;
            InitialConverter = initCon;
            AdderFunction = adder;
            WeightingFunction = weighting;

            LayerIndex = layer;
            OwnerNet = owner;
        }

        public TOutput Calc()
        {
            if (LayerIndex == 0)
            {
                TActivator temp = InitialConverter(
                    WeightingFunction(
                        OwnerNet.Input.First(), 
                        Weights.First()
                        )
                    );
                for (int i = 1; i < OwnerNet.Input.Length; i++)
                    temp = AdderFunction(temp, WeightingFunction(OwnerNet.Input[i], Weights[i]));
                return ActivationFunction(temp);
            }
            else
            {
                TActivator temp = InitialConverter(
                    WeightingFunction(
                        InterchangeConverter(
                            OwnerNet.Neurons[LayerIndex - 1].First().Calc()
                            ), 
                        Weights.First()
                        )
                    );
                for (int i = 1; i < Weights.Count; i++)
                    temp = AdderFunction(temp, WeightingFunction(InterchangeConverter(OwnerNet.Neurons[LayerIndex - 1][i].Calc()), Weights[i]));
                return ActivationFunction(temp);
            }
        }
    }
}
