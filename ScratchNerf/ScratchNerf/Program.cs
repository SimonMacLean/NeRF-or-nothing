using System.Numerics;
using AcceleratedNeRFUtils;

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
            BinDataset binDataset = new("C:\\Users\\simon\\Desktop\\mipnerf-main\\train_data.bin");
            AcceleratedMipNeRF model = new();
            AcceleratedAdamOptimizer optimizer = new(model.GetLayerSizes());
            AcceleratedGradientCalculator gradientCalculator = new(BinDataset.BatchSize);
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
                (Rays rays, Vector3[] pixels) batch = binDataset.Next();
                float lr = learningRateFn(step);

                TrainStep(model, optimizer, gradientCalculator, batch, lr);

                if (step % Config.PrintEvery == 0)
                {
                    Console.WriteLine($@"Step {step}/{Config.MaxSteps}");
                }
            }
        }

        static unsafe void TrainStep(AcceleratedMipNeRF model, AcceleratedAdamOptimizer optimizer, AcceleratedGradientCalculator gradientCalculator, (Rays rays, Vector3[] pixels) batch, float learningRate)
        {
            float*[] grad = model.GetGradient(batch.rays.Origins, batch.rays.Directions, batch.rays.Radii,
                batch.rays.Nears, batch.rays.Fars, batch.rays.LossMults,
                (inputptr, level, lossMultSum, lossMults) =>
                    gradientCalculator.get_output_gradient(inputptr, batch.pixels, lossMults, lossMultSum, level));
            float*[] variables = model.mlp.allParams;
            optimizer.step(variables, grad, learningRate);
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