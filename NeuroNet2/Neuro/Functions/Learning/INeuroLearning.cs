using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuroNet2.Neuro.Mesh;

namespace NeuroNet2.Neuro.Functions.Learning
{
    interface INeuroLearning<T>
    {
        double LearnIteration(FullMesh<T> net, T[][] input, T[][] output, double switchThreshold);
    }
}
