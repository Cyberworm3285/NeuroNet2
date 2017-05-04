using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Functions.Learning;

namespace NeuroNet2.Neuro.Mesh
{
    class FullMesh<T>
    {
        public List<List<Neuron<T>>> Neurons { get; private set; }
        public T[] Input { get; private set; }
        public double LastError { get; private set; }
        public Func<T, T ,double> ErrorFunction { get; set; } 

        public FullMesh(int inputCounts, int[] neuroCounts,Func<T,T,double> errorFunction, Func<T,T> activationFunction, Func<T,T,T> adderFunction, Func<T,T,T> weightingFunction, T defaultWeight)
        {
            ErrorFunction = errorFunction;
            LastError = double.MaxValue;
            Neurons = new List<List<Neuron<T>>>();
            Input = new T[inputCounts];
            for (int i = 0; i < neuroCounts.Length; i++)
            {
                Neurons.Add(new List<Neuron<T>>());
                for (int j = 0; j < neuroCounts[i]; j++)
                    AddNeuronInLayer(
                        new Neuron<T>(activationFunction, adderFunction, weightingFunction, i, this),
                        i,
                        defaultWeight
                        );
            }
        }

        public void SetInput(params T[] newInput)
        {
            if (Input.Length != newInput.Length)
                throw new Exception();
            Input = newInput;
        }

        public void AddNeuronInLayer(Neuron<T> n, int layerIndex, T defaultWeight)
        {
            if (layerIndex < 0 || layerIndex > Neurons.Count)
                throw new Exception();
            Neurons[layerIndex].Add(n);
            n.Weights = new List<T>();
            //dem neuen Neuron alle vorherigen Verbindungen
            for (int i = 0; i < ( (layerIndex == 0) ? Input.Length : Neurons[layerIndex - 1].Count ); i++)
                n.Weights.Add(defaultWeight);
            //den Neuronen in der nächsten Schicht eine neue Verbindung
            if (layerIndex < Neurons.Count - 1)
                Neurons[layerIndex + 1].ForEach(nn => nn.Weights.Add(defaultWeight));
        }

        public T[] Calc()
        {
            return Neurons.Last().Select(n => n.Calc()).ToArray();
        }

        public T[] Calc(T[] input)
        {
            SetInput(input);
            return Calc();
        }

        public double GetError(T[] calculated, T[] expected)
        {
            if (calculated.Length != expected.Length)
                throw new Exception();
            double result = 0.0;
            for (int i = 0; i < calculated.Length; i++)
                result += ErrorFunction(calculated[i], expected[i]);
            return result;
        }

        public double GetError(T[][] calculated, T[][] expected)
        {
            if (calculated.Length != expected.Length)
                throw new Exception();
            double result = 0.0;
            for (int i = 0; i < calculated.Length; i++)
                result += GetError(calculated[i], expected[i]);
            return result;
        }

        public double LearnIteration(INeuroLearning<T> learning, T[][] input, T[][] output, double switchThreshold)
        {
            LastError = learning.LearnIteration(this, input, output, switchThreshold);
            return LastError;
        }

        public void SetActivationFunction(Func<T,T> function)
        {
            Neurons.ForEach(n => n.ForEach(nn => nn.ActivationFunction = function));
        }

        public void SetActivationFunctionToLayer(int layerIndex, Func<T,T> function)
        {
            if (layerIndex < 0 || layerIndex > Neurons.Count)
                throw new Exception();
            Neurons[layerIndex].ForEach(n => n.ActivationFunction = function);
        }
    }
}
