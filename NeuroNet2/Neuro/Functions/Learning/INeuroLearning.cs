using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Mesh;

namespace NeuroNet2.Neuro.Functions.Learning
{
    interface INeuroLearning<TInput, TOutput, TWeight, TActivator>
    {
        double LearnIteration(FullMesh<TInput, TOutput, TWeight, TActivator> net, TInput[][] input, TOutput[][] output, double switchThreshold);
    }
}
