using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Mesh;
using NeuroNet2.Neuro.Functions.Learning;
using static NeuroNet2.Neuro.Functions.Activation.ActivationFunctions;
using NeuroNet2.Global;

namespace NeuroNet2
{
    class Program
    {
        static void Main(string[] args)
        {
            B();
        }

        static void A()
        {
            FullMesh<double> mesh = new FullMesh<double>(
            inputCounts: 2,
            neuroCounts: new int[] { 5, 5, 5, 1 },
            errorFunction: (a, b) => Math.Abs(a - b),
            activationFunction: d => d,
            adderFunction: (a, b) => a + b,
            weightingFunction: (a, b) => a * b,
            defaultWeight: 0.5
            );

            double[][] input = new double[][]
            {
                new[] { 0.0, 0.0 },
                new[] { 0.0, 1.0 },
                new[] { 2.0, 4.0 },
                new[] { 10.0, 1.0 },
            };

            double[][] output = new double[][]
{
                new[] { 0.0},
                new[] { 1.0},
                new[] { 6.0},
                new[] { 11.0},
            };

            for (int i = 0; i < 20; i++)
                mesh.LearnIteration(new DeepRandom<double>(250, d => d + GlobalRandom.Get.NextDouble() * 0.4 - 0.2) { PrintProgress = true }, input, output, -0.0001);
            for (int i = 0; i < 10000 && mesh.LastError > 0.00001; i++)
                Console.WriteLine(mesh.LearnIteration(new Evolution<double>(d => d + GlobalRandom.Get.NextDouble() * 0.4 - 0.2), input, output, -0.00001));
            //mesh.LearnIteration(new BruteForce<double>(d => d + GlobalRandom.Get.NextDouble() * 0.4 - 0.2) { PrintProgress = true }, input, output, 0.01);

            Array.ForEach(input, d => Console.WriteLine(Format(d) + " : " + Format(mesh.Calc(d))));

            Console.ReadKey();
        }

        static void B()
        {
            FullMesh<bool> mesh = new FullMesh<bool>(
                inputCounts : 2,
                neuroCounts : new int[] { 10, 10, 1 },
                errorFunction : (a,b) => (a != b) ? 1 : 0,
                activationFunction : b => b,
                adderFunction : (a, b) => a || b,
                weightingFunction : (a,b) => a || b,
                defaultWeight: false
                );

            Func<bool, bool, bool>[] funcs = new Func<bool, bool, bool>[]
            {
                (a,b) => !(a&&b),
                (a,b) => a&&b,
                (a,b) => !(a||b),
                (a,b) => a||b
            };

            mesh.Neurons.ForEach(n => n.ForEach(nn => nn.WeightingFunction = funcs[GlobalRandom.Get.Next(0, funcs.Length)]));

            bool[][] input = new bool[][]
            {
                new[] { false, false },
                new[] { false, true },
                new[] { true, false },
                new[] { true, true },
            };

            bool[][] output = new bool[][]
            {
                new[] { false },
                new[] { true },
                new[] { true },
                new[] { false },
            };


            mesh.LearnIteration(new BruteForce<bool>(b => !b) { PrintProgress = true }, input, output, 0.01);
            //for (int i = 0; i < 20; i++)
            //    mesh.LearnIteration(new DeepRandom<bool>(250, d => !d) { PrintProgress = true }, input, output, -0.0001);
            //for (int i = 0; i < 10000 && mesh.LastError > 0.00001; i++)
            //    Console.WriteLine(mesh.LearnIteration(new Evolution<bool>(b => !b), input, output, -0.00001));

            Array.ForEach(input, d => Console.WriteLine(Format(d) + " : " + Format(mesh.Calc(d))));

            Console.ReadKey();
        }

        static string Format<T>(T[] d) => "[" + string.Join(",", d.Select(dd => dd.ToString())) + "]";
    }
}
