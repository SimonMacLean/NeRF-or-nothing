using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ScratchNerf
{
    public class StatsUtil
    {
        public float loss;
        public float[] losses;
        public float weightL2;
        public float psnr;
        public float[] psnrs;
        public float gradNorm;
        public float gradAbsMax;
        public float graphNormClipped;
    }
}
