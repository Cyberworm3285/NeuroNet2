using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using static System.Math;

namespace NeuroNet2.Neuro.Functions.Activation
{
    static class ActivationFunctions
    {
        public static double TangensHyperbolicus(double d) => 1 - (2 / (Pow(E, 2 * d) + 1));
    }
}
