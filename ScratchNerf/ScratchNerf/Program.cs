using System.Data;
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
            Train();
        }

        static void Train()
        {
            Dataset dataset = DatasetFactory.CreateDataset(Split.Train, Flags.dataDir);
            Random rng = new();
            (MipNerfModel model, float[] variables) = MipNerfModel.ConstructMipNerf(dataset.Peek().rays);

            int numParams = variables.Length;
            Console.WriteLine($"Number of parameters being optimized: {numParams}");

            AdamOptimizer optimizer = new AdamOptimizer(numParams, Config.LrInit);
            TrainState state = new TrainState(optimizer);

            Func<int, float> learningRateFn = step => MathHelpers.LearningRateDecay(
                step,
                Config.LrInit,
                Config.LrFinal,
                Config.MaxSteps,
                Config.LrDelaySteps,
                Config.LrDelayMult
            );

            for (int step = 1; step <= Config.MaxSteps; step++)
            {
                var batch = dataset.Next();
                float lr = learningRateFn(step);

                StatsUtil stats = TrainStep(model, rng, state, batch, lr);

                if (step % Config.PrintEvery == 0)
                {
                    Console.WriteLine($@"Step {step}/{Config.MaxSteps}: loss={stats.loss:F4}, psnr={stats.psnr:F2}, lr={lr:E2}");
                }
            }
        }

        static StatsUtil TrainStep(MipNerfModel model, Random rng, TrainState state, (Ray[] rays, Vector3[] pixels) batch, float learningRate)
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
            (StatsUtil stats, float[] gradient) = model.GetGradient(batch.rays, Config.Randomized, Config.WhiteBkgd, GetGradient, (x) => LossFn(x, batch));
            float[] variables = model.mlp.allParams;
            for (int i = 0; i < variables.Length; i++)
            {
                gradient[i] += Config.WeightDecayMult * variables[i];
            }
            if(Config.GradMaxVal > 0) gradient = gradient.Select((x) => Math.Clamp(x, -Config.GradMaxVal, Config.GradMaxVal)).ToArray();
            float gradMaxAbs = gradient.Max(Math.Abs);
            float gradNorm = MathF.Sqrt(gradient.Select((x) => x * x).Sum());
            if(Config.GradMaxNorm > 0 && gradNorm > Config.GradMaxNorm)
            {
                gradient = gradient.Select((x) => x * Config.GradMaxNorm / gradNorm).ToArray();
            }
            float gradNormClipped = MathF.Sqrt(gradient.Select((x) => x * x).Sum());
            state.Optimizer.Step(variables, gradient);
            float[] psnrs = stats.losses.Select(MathHelpers.MseToPsnr).ToArray();
            stats = new StatsUtil
            {
                loss = stats.loss,
                losses = stats.losses,
                weightL2 = stats.weightL2,
                gradAbsMax = gradMaxAbs,
                gradNorm = gradNorm,
                gradNormClipped = gradNormClipped,
                psnrs = psnrs,
                psnr = psnrs[^1]
            };
            return stats;
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

        static (float, StatsUtil) LossFn((Vector3 CompositeRgb, float Distance, float Accumulation)[,] ret,
            (Ray[] rays, Vector3[] pixels) batch)
        {
                        float[] mask = batch.rays.Select((ray) => ray.LossMult).ToArray();
            if (Config.DisableMultiscaleLoss) mask = mask.Select((x) => 1f).ToArray();
            float[] losses = new float[ret.GetLength(0)];
            for (int level = 0; level < ret.GetLength(0); level++)
            {
                float totalLoss = 0;
                float totalMask = mask.Sum();

                for (int i = 0; i < batch.rays.Length; i++) totalLoss += mask[i] * (ret[level, i].CompositeRgb - batch.pixels[i]).LengthSquared();

                losses[level] = totalLoss / totalMask;
            }
            float loss = losses[..^1].Sum() * Config.CoarseLossMult + losses[^1];
            StatsUtil stats = new()
            {
                loss = loss,
                losses = losses,
                weightL2 = 0,
            };
            return (loss, stats);
        }
    }
}