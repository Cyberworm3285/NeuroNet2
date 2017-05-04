using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroNet2.Global
{
    static class GlobalRandom
    {
        private static Random random = new Random();
        public static Random Get => random;
    }
}
