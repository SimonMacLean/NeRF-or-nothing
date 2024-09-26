using System.Numerics;
using System.Reflection;
using System.Threading.Tasks;

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
            int numLevels = model.NumLevels;
            int numRays = batch.rays.Length;
            float[] mask = batch.rays.Select((ray) => ray.LossMult).ToArray();
            Vector3[,] GetGradient(
                (Vector3 CompositeRgb, float Distance, float Accumulation)[,] returnedValue)
            {
                Vector3[,] gradient =
                    new Vector3[numLevels,
                        returnedValue.GetLength(1)];

                for (int level = 0; level < numLevels; level++)
                {
                    float totalMask = mask.Sum();
                    for (int i = 0; i < returnedValue.GetLength(1); i++)
                    {
                        Vector3 diffRgb = returnedValue[level, i].CompositeRgb - batch.pixels[i];
                        float maskFactor = mask[i] / totalMask;
                        gradient[level, i] = 2 * maskFactor * diffRgb;
                        if (level < numLevels - 1)
                        {
                            gradient[level, i] *= Config.CoarseLossMult;
                        }
                    }
                }

                return gradient;
            }
            float[] gradient = model.GetGradient(batch.rays, Config.Randomized, Config.WhiteBkgd, GetGradient);


        }
        static (float, StatsUtil) LossFn(MipNerfModel model, Random rng, TrainState state, (Ray[] rays, Vector3[] pixels) batch, float learningRate)
        {
            float weightL2 = Config.WeightDecayMult * model.mlp.allParams.Average((z) => z * z);
            (Vector3 CompositeRgb, float Distance, float Accumulation)[,] ret = model.Call(batch.rays, Config.Randomized, Config.WhiteBkgd);
            float[] mask = batch.rays.Select((ray) => ray.LossMult).ToArray();
            if (Config.DisableMultiscaleLoss) mask = mask.Select((x) => 1f).ToArray();
            float[] losses = new float[model.NumLevels];
            for (int level = 0; level < model.NumLevels; level++)
            {
                float totalLoss = 0;
                float totalMask = mask.Sum();

                for (int i = 0; i < batch.rays.Length; i++) totalLoss += mask[i] * (ret[level, i].CompositeRgb - batch.pixels[i]).LengthSquared();

                losses[level] = totalLoss / totalMask;
            }
            float loss = losses[..^1].Sum() * Config.CoarseLossMult + losses[^1] + weightL2;
            StatsUtil stats = new()
            {
                loss = loss,
                losses = losses,
                weightL2 = weightL2,
            };
            return (loss, stats);
        }

        static float[] CalculateGradient(MipNerfModel model, Random rng, TrainState state,
            (Ray[] rays, Vector3[] pixels) batch, float learningRate)
        {
            (Vector3 CompositeRgb, float Distance, float Accumulation)[,] ret = model.Call(batch.rays, Config.Randomized, Config.WhiteBkgd);
            int numLevels = model.NumLevels;
            int numRays = batch.rays.Length;
            float[,] retGradient = new float[numLevels]
        }
    }
}