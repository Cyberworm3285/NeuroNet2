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
            A();
        }

        static void A()
        {
            FullMesh<double> mesh = new FullMesh<double>(
            inputCounts: 2,
            neuroCounts: new int[] { 10, 1 },
            errorFunction: (a, b) => Math.Abs(a - b),
            activationFunction: TangensHyperbolicus,
            adderFunction: (a, b) => a + b,
            weightingFunction: (a, b) => a * b,
            defaultWeight: 0.5
            );

            Func<double, double, double>[] funcs = new Func<double, double, double>[]
            {
                (a,b) => a+b,
                (a,b) => a-b,
                (a,b) => a*b,
                (a,b) => a%b,
            };

            double[][] input = new double[][]
            {
                new[] { 0.0, 0.0 },
                new[] { 0.0, 1.0 },
                new[] { 1.0, 0.0 },
                new[] { 1.0, 1.0 },
            };

            double[][] output = new double[][]
{
                new[] { 1.0},
                new[] { 1.0},
                new[] { 0.0},
                new[] { 1.0},
            };

            //for (int i = 0; i < 20; i++)
            //    mesh.LearnIteration(new DeepRandom<double>(250, d => d + GlobalRandom.Get.NextDouble() * 0.4 - 0.2) { PrintProgress = true }, input, output, -0.0001);
            //for (int i = 0; i < 10000 && mesh.LastError > 0.00001; i++)
            //    mesh.LearnIteration(new Evolution<double>(d => d + GlobalRandom.Get.NextDouble() * 0.4 - 0.2), input, output, -0.00001);
            mesh.LearnIteration(new BruteForce<double>(d => d + GlobalRandom.Get.NextDouble() * 0.4 - 0.2, () => funcs[GlobalRandom.Get.Next(0, funcs.Length)]) { PrintProgress = true }, input, output, 0.01);

            Array.ForEach(input, d => Console.WriteLine(Format(d) + " : " + Format(mesh.Calc(d))));

            while(true)
                Console.WriteLine(Format(mesh.Calc(Console.ReadLine().Split(' ').Select(s => double.Parse(s)).ToArray())));
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
                (a,b) => !(!a&&b),
                (a,b) => !(a&&!b),
                (a,b) => !(!a&&!b),
                (a,b) => a&&b,
                (a,b) => !a&&b,
                (a,b) => a&&!b,
                (a,b) => !a&&!b,
                (a,b) => !(a||b),
                (a,b) => !(!a||b),
                (a,b) => !(a||!b),
                (a,b) => !(!a||!b),
                (a,b) => a||b,
                (a,b) => !a||b,
                (a,b) => a||!b,
                (a,b) => !a||!b,
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
                new[] { true },
                new[] { true },
                new[] { false },
                new[] { true },
            };


            mesh.LearnIteration(new BruteForce<bool>(b => b, () => funcs[GlobalRandom.Get.Next(0, funcs.Length)]) { PrintProgress = true }, input, output, 0.01);

            Array.ForEach(input, d => Console.WriteLine(Format(d) + " : " + Format(mesh.Calc(d))));

            Console.ReadKey();
        }

        static string Format<T>(T[] d) => "[" + string.Join(",", d.Select(dd => dd.ToString())) + "]";
    }
}
