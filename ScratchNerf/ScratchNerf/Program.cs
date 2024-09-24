using System.Numerics;

namespace ScratchNerf
{
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            // To customize application configuration such as set high DPI settings or default font,
            // see https://aka.ms/applicationconfiguration.
            //ApplicationConfiguration.Initialize();
            //Application.Run(new Form1());

        }

        static void Train()
        {
            Ray[][] dataset = [];
            MipNerfModel mipNerfModel = new MipNerfModel();
            float[] variables = mipNerfModel.mlp.allParams;
            int numParams = variables.Length;
            Console.WriteLine("Number of parameters being optimized: " + numParams);
        }

        static void TrainStep(MipNerfModel model, Random rng, TrainState state, (Ray[] rays, Vector3[] pixels) batch, float learningRate)
        {
            float LossFn(float[] variables)
            {
                float weightL2 = Config.WeightDecayMult * variables.Average((z) => z * z);
                MipNerfModel model = new();
                model.mlp.allParams = variables;
                (Vector3 CompositeRgb, float Distance, float Accumulation)[,] ret = model.Call(batch.rays, Config.Randomized, Config.WhiteBkgd);
                float[] mask = batch.rays.Select((ray) => ray.LossMult).ToArray();
                if(Config.DisableMultiscaleLoss) mask = mask.Select((x) => 1f).ToArray();
                float[] losses = new float[model.NumLevels];

                for (int level = 0; level < model.NumLevels; level++)
                {
                    float loss = 0;
                    float totalMask = 0;

                    for (int i = 0; i < batch.rays.Length; i++)
                    {
                        Vector3 diff = ret[level, i].Rgb - pixels[i];
                        float squaredError = diff.X * diff.X + diff.Y * diff.Y + diff.Z * diff.Z;
                        loss += rays[i].LossMult * squaredError;
                        totalMask += rays[i].LossMult;
                    }

                    losses.Add(loss / totalMask);
                }
            }
        }

    }
}