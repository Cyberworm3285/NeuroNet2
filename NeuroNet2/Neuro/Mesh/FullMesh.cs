using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Functions.Learning;

namespace NeuroNet2.Neuro.Mesh
{
    class FullMesh<TInput, TOutput, TWeight, TActivator>
    {
        public List<List<Neuron<TInput, TOutput, TWeight, TActivator>>> Neurons { get; private set; }
        public TInput[] Input { get; private set; }
        public double LastError { get; private set; }
        public Func<TOutput, TOutput ,double> ErrorFunction { get; set; } 

        public FullMesh(
            int inputCounts, 
            int[] neuroCounts,
            Func<TOutput, TOutput, 
            double> errorFunction, 
            Func<TActivator, TOutput> activationFunction, 
            Func<TActivator, TInput, TActivator> adderFunction, 
            Func<TInput, TWeight, TInput> weightingFunction,
            Converter<TOutput, TInput> interchangeCon,
            Converter<TInput, TActivator> initialCon,
            TWeight defaultWeight)
        {
            ErrorFunction = errorFunction;
            LastError = double.MaxValue;
            Neurons = new List<List<Neuron<TInput, TOutput, TWeight, TActivator>>>();
            Input = new TInput[inputCounts];
            for (int i = 0; i < neuroCounts.Length; i++)
            {
                Neurons.Add(new List<Neuron<TInput, TOutput, TWeight, TActivator>>());
                for (int j = 0; j < neuroCounts[i]; j++)
                    AddNeuronInLayer(
                        new Neuron<TInput, TOutput, TWeight, TActivator>(interchangeCon, activationFunction, initialCon, adderFunction, weightingFunction, i, this),
                        i,
                        defaultWeight
                        );
            }
        }

        public void SetInput(params TInput[] newInput)
        {
            if (Input.Length != newInput.Length)
                throw new Exception();
            Input = newInput;
        }

        public void AddNeuronInLayer(Neuron<TInput, TOutput, TWeight, TActivator> n, int layerIndex, TWeight defaultWeight)
        {
            if (layerIndex < 0 || layerIndex > Neurons.Count)
                throw new Exception();
            Neurons[layerIndex].Add(n);
            n.Weights = new List<TWeight>();
            //dem neuen Neuron alle vorherigen Verbindungen
            for (int i = 0; i < ( (layerIndex == 0) ? Input.Length : Neurons[layerIndex - 1].Count ); i++)
                n.Weights.Add(defaultWeight);
            //den Neuronen in der nächsten Schicht eine neue Verbindung
            if (layerIndex < Neurons.Count - 1)
                Neurons[layerIndex + 1].ForEach(nn => nn.Weights.Add(defaultWeight));
        }

        public TOutput[] Calc()
        {
            return Neurons.Last().Select(n => n.Calc()).ToArray();
        }

        public TOutput[] Calc(TInput[] input)
        {
            SetInput(input);
            return Calc();
        }

        public double GetError(TOutput[] calculated, TOutput[] expected)
        {
            if (calculated.Length != expected.Length)
                throw new Exception();
            double result = 0.0;
            for (int i = 0; i < calculated.Length; i++)
                result += ErrorFunction(calculated[i], expected[i]);
            return result;
        }

        public double GetError(TOutput[][] calculated, TOutput[][] expected)
        {
            if (calculated.Length != expected.Length)
                throw new Exception();
            double result = 0.0;
            for (int i = 0; i < calculated.Length; i++)
                result += GetError(calculated[i], expected[i]);
            return result;
        }

        public double LearnIteration(INeuroLearning<TInput, TOutput, TWeight, TActivator> learning, TInput[][] input, TOutput[][] output, double switchThreshold)
        {
            LastError = learning.LearnIteration(this, input, output, switchThreshold);
            return LastError;
        }

        public void SetActivationFunction(Func<TActivator,TOutput> function)
        {
            Neurons.ForEach(n => n.ForEach(nn => nn.ActivationFunction = function));
        }

        public void SetActivationFunctionToLayer(int layerIndex, Func<TActivator,TOutput> function)
        {
            if (layerIndex < 0 || layerIndex > Neurons.Count)
                throw new Exception();
            Neurons[layerIndex].ForEach(n => n.ActivationFunction = function);
        }
    }
}
