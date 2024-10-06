using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ScratchNerf
{
    using System;
    using System.Collections.Generic;
    using System.Numerics;
    using System.Reflection;
    using System.Runtime.InteropServices;

    public class MipNerfModel
    {
        // Fields and Properties
        public int NumSamples { get; set; } = 128;  // The number of samples per level
        public int NumLevels { get; set; } = 2;     // The number of sampling levels
        public float ResamplePadding { get; set; } = 0.01f;  // Dirichlet/alpha "padding" on the histogram
        public bool StopLevelGrad { get; set; } = true;  // If True, don't backprop across levels
        public bool LinDisp { get; set; } = false;   // If True, sample linearly in disparity, not in depth
        public MipHelpers.RayShape RayShape { get; set; } = MipHelpers.RayShape.Conical;  // The shape of cast rays ("cone" or "cylinder")
        public int MinDegPoint { get; set; } = 0;    // Min degree of positional encoding for 3D points
        public int MaxDegPoint { get; set; } = 16;   // Max degree of positional encoding for 3D points
        public int DegView { get; set; } = 4;        // Degree of positional encoding for view directions
        public Func<float, float> DensityActivation { get; set; } = x => MathF.Log(1 + MathF.Exp(x));  // Density activation (softplus)
        public float DensityBias { get; set; } = -1f;   // The shift added to raw densities pre-activation
        public Func<float, float> RgbActivation { get; set; } = x => 1 / (1 + MathF.Exp(-x));  // The RGB activation (sigmoid)
        public float RgbPadding { get; set; } = 0.001f;  // Padding added to the RGB outputs
        public Func<float, float> DensityActivationGradient { get; set; } = x => 1 / (1 + MathF.Exp(-x));  // The gradient of the density activation
        public Func<float, float> RgbActivationGradient { get; set; } = x =>
        {
            float exp = MathF.Exp(x);
            return exp / (1 + exp) / (1 + exp);
        };  // The gradient of the RGB activation

        Random rng = new();
        public MLP mlp;

        public MipNerfModel() => mlp = new(MaxDegPoint - MinDegPoint, DegView);

        // Methods
        public (Vector3 CompositeRgb, float Distance, float Accumulation)[,] Call(Ray[] rays, bool randomized, bool whiteBackground)
        {
            int numRays = rays.Length;
            (Vector3 CompositeRgb, float Distance, float Accumulation)[,] results = new (Vector3 CompositeRgb, float Distance, float Accumulation)[NumLevels, numRays];
            float[] weights = [];
            for (int iLevel = 0; iLevel < NumLevels; iLevel++)
            {
                (float[] tVals, (Vector3 mean, Matrix3x3 covariance)[] samples)[] res = new (float[] tVals, (Vector3 mean, Matrix3x3 covariance)[] samples)[numRays];
                if (iLevel == 0)
                {
                    for (int i = 0; i < rays.Length; i++)
                    {
                        Ray r = rays[i];
                        res[i] = MipHelpers.SampleAlongRay(rng, r.Origin, r.Direction, r.Radius, NumSamples, r.Near, r.Far, randomized, LinDisp, RayShape);
                    }
                }
                else
                {
                    for(int i = 0; i < rays.Length; i++)
                    {
                        Ray r = rays[i];
                        (float[] tVals, (Vector3 mean, Matrix3x3 covariance)[] samples) prevRes = res[i];
                        res[i] = MipHelpers.ResampleAlongRay(rng, r.Origin, r.Direction, r.Radius, prevRes.tVals,
                            weights, randomized, RayShape, StopLevelGrad, ResamplePadding);
                    }
                }
                Vector3[,][] samplesEnc = new Vector3[numRays, NumSamples + 1 - iLevel][];
                Vector3[][] dirsEnc = new Vector3[numRays][];
                for (int iRay = 0; iRay < res.Length; iRay++)
                {
                    for (int iSample = 0; iSample < res[iRay].samples.Length; iSample++)
                        samplesEnc[iRay, iSample] =
                            MipHelpers.IntegratedPositionalEncoding(res[iRay].samples[iSample], MinDegPoint,
                                MaxDegPoint);
                    dirsEnc[iRay] = MipHelpers.PositionalEncoding(rays[iRay].Direction, 0, DegView);
                }
                (Vector3 rawRgb, float rawDensity)[,] rawOutput = new (Vector3 rgb, float density)[numRays, NumSamples + 1 - iLevel];
                for (int iRay = 0; iRay < res.Length; iRay++)
                for (int iSample = 0; iSample < res[iRay].samples.Length; iSample++) rawOutput[iRay, iSample] = mlp.Call(samplesEnc[iRay, iSample], dirsEnc[iRay]);
                (Vector3 rgb, float density)[,] finalOutput = new (Vector3 rgb, float density)[numRays, NumSamples + 1 - iLevel];
                for (int iRay = 0; iRay < res.Length; iRay++)
                    for (int iSample = 0; iSample < res[iRay].samples.Length; iSample++)
                    {
                        Vector3 rawRgb = rawOutput[iRay, iSample].rawRgb;
                        float rawDensity = rawOutput[iRay, iSample].rawDensity;
                        Vector3 rgb = new(RgbActivation(rawRgb.X), RgbActivation(rawRgb.Y), RgbActivation(rawRgb.Z));
                        rgb = rgb * (1 + 2 * RgbPadding) - Vector3.One*RgbPadding;
                        float density = DensityActivation(rawDensity + DensityBias);
                        finalOutput[iRay, iSample] = (rgb, density);
                    }
                for (int i = 0; i < numRays; i++)
                {
                    (Vector3 rawRgb, float rawDensity)[] rayOutput = new (Vector3 rawRgb, float rawDensity)[NumSamples + 1 - iLevel];
                    for (int j = 0; j < NumSamples + 1 - iLevel; j++) rayOutput[j] = finalOutput[i, j];
                    (Vector3 compositeRgb, float distance, float accumulation, float[] weights) withWeights = MipHelpers.VolumetricRendering(rayOutput, res[i].tVals, rays[i].Direction, whiteBackground);
                    results[iLevel, i] = (withWeights.compositeRgb, withWeights.distance, withWeights.accumulation);
                    weights = withWeights.weights;
                }
            }

            return results;
        }

        public (StatsUtil, float[]) GetGradient(Ray[] rays, bool randomized, bool whiteBackground,
            Func<(Vector3 CompositeRgb, float Distance, float Accumulation)[,] , Vector3[,]> getReturnGradient, Func<(Vector3 CompositeRgb, float Distance, float Accumulation)[,], (float loss, StatsUtil stats)> LossFn)
        {
            int numRays = rays.Length;
            (Vector3 CompositeRgb, float Distance, float Accumulation)[,] results =
                new (Vector3 CompositeRgb, float Distance, float Accumulation)[NumLevels, numRays];
            float[,][] weights = new float[NumLevels, numRays][];
            float[,][] transmittance = new float[NumLevels, numRays][];
            float[,][] alpha = new float[NumLevels, numRays][];
            (Vector3 rgb, float density)[,][] finalOutput = new (Vector3 rgb, float density)[NumLevels,numRays][];
            (Vector3 rawRgb, float rawDensity)[,][] rawOutput = new (Vector3 rawRgb, float rawDensity)[NumLevels,numRays][];
            (float[] tVals, (Vector3 mean, Matrix3x3 covariance)[] samples)[,] res =
                new (float[] tVals, (Vector3 mean, Matrix3x3 covariance)[] samples)[NumLevels, numRays];
            float[,][][][] inputs = new float[NumLevels, numRays][][][];
            for (int iLevel = 0; iLevel < NumLevels; iLevel++)
            {
                if (iLevel == 0)
                {
                    for (int iRay = 0; iRay < numRays; iRay++)
                    {
                        Ray r = rays[iRay];
                        res[iLevel, iRay] = MipHelpers.SampleAlongRay(rng, r.Origin, r.Direction, r.Radius, NumSamples,
                            r.Near, r.Far, randomized, LinDisp, RayShape);
                    }
                }
                else
                {
                    for (int iRay = 0; iRay < numRays; iRay++)
                    {
                        Ray r = rays[iRay];
                        res[iLevel, iRay] = MipHelpers.ResampleAlongRay(rng, r.Origin, r.Direction, r.Radius,
                            res[iLevel - 1, iRay].tVals,
                            weights[iLevel - 1, iRay], randomized, RayShape, StopLevelGrad, ResamplePadding);
                    }
                }

                Vector3[,][] samplesEnc = new Vector3[numRays, NumSamples + 1 - iLevel][];
                Vector3[][] dirsEnc = new Vector3[numRays][];
                for (int iRay = 0; iRay < numRays; iRay++)
                {
                    for (int iSample = 0; iSample < res[iLevel, iRay].samples.Length; iSample++)
                        samplesEnc[iRay, iSample] =
                            MipHelpers.IntegratedPositionalEncoding(res[iLevel, iRay].samples[iSample], MinDegPoint,
                                MaxDegPoint);
                    dirsEnc[iRay] = MipHelpers.PositionalEncoding(rays[iRay].Direction, 0, DegView);
                    inputs[iLevel, iRay] = new float[NumSamples + 1 - iLevel][][];
                    rawOutput[iLevel, iRay] = new (Vector3 rgb, float density)[NumSamples + 1 - iLevel];
                    finalOutput[iLevel, iRay] = new (Vector3 rgb, float density)[NumSamples + 1 - iLevel];
                    for (int iSample = 0; iSample < res[iLevel, iRay].samples.Length; iSample++)
                    {
                        (Vector3 rawRgb, float rawDensity, inputs[iLevel, iRay][iSample]) = mlp.CallCached(samplesEnc[iRay, iSample], dirsEnc[iRay]);
                        rawOutput[iLevel, iRay][iSample] = (rawRgb, rawDensity);
                        Vector3 rgb = new Vector3(RgbActivation(rawRgb.X), RgbActivation(rawRgb.Y), RgbActivation(rawRgb.Z)) * (1 + 2 * RgbPadding) - Vector3.One * RgbPadding;
                        float density = DensityActivation(rawDensity + DensityBias);
                        finalOutput[iLevel, iRay][iSample] = (rgb, density);
                    }
                    (results[iLevel, iRay].CompositeRgb, results[iLevel, iRay].Distance, results[iLevel, iRay].Accumulation, alpha[iLevel, iRay], transmittance[iLevel, iRay],
                        weights[iLevel, iRay]) = MipHelpers.CachedVolumetricRendering(finalOutput[iLevel, iRay],
                        res[iLevel, iRay].tVals, rays[iRay].Direction, whiteBackground);
                    if (iRay % 16 == 0) Console.Write("█");
                }
            }
            Console.WriteLine();
            Vector3[,] rgbGradient = getReturnGradient(results);
            (float loss, StatsUtil stats) = LossFn(results);
            Console.WriteLine($"Loss: {loss}");
            (Vector3 rgbGradient, float densityGradient)[,][] finalOutputGradient = new (Vector3 rgbGradient, float densityGradient)[NumLevels,numRays][];
            int numParams = mlp.allParams.Length;
            float[] summedParamsGradient = new float[numParams];
            for (int iLevel = NumLevels - 1; iLevel >= 0; iLevel--)
            {
                for (int iRay = 0; iRay < numRays; iRay++)
                {
                    finalOutputGradient[iLevel, iRay] = MipHelpers.VolumetricRenderingGradient(rgbGradient[iLevel, iRay],
                        results[iLevel, iRay].Distance, results[iLevel, iRay].Accumulation, alpha[iLevel, iRay],
                        transmittance[iLevel, iRay], weights[iLevel, iRay], res[iLevel, iRay].tVals, rays[iRay].Direction,
                        whiteBackground);
                    for (int iSample = 0; iSample < res[iLevel, iRay].samples.Length; iSample++)
                    {
                        (Vector3 rawRgb, float rawDensity) = rawOutput[iLevel, iRay][iSample];
                        (Vector3 rgbGrad, float densityGrad) = finalOutputGradient[iLevel, iRay][iSample];

                        // Apply activation gradients
                        Vector3 rawRgbGrad = new Vector3(
                            rgbGrad.X * RgbActivationGradient(rawRgb.X),
                            rgbGrad.Y * RgbActivationGradient(rawRgb.Y),
                            rgbGrad.Z * RgbActivationGradient(rawRgb.Z)
                        ) * (1 + 2 * RgbPadding);
                        float rawDensityGrad = densityGrad * DensityActivationGradient(rawDensity + DensityBias);
                        float[] currentParamsGradient = mlp.GetGradient(inputs[iLevel, iRay][iSample], rawRgbGrad, rawDensityGrad);
                        for (int i = 0; i < numParams; i++)
                            summedParamsGradient[i] += currentParamsGradient[i];
                    }
                    if (iRay % 16 == 0) Console.Write("█");
                }
            }

            Console.WriteLine();
            return (stats, summedParamsGradient);
        }

        public static (MipNerfModel, float[]) ConstructMipNerf()
        {
            MipNerfModel model = new();
            return (model, model.mlp.allParams);
        }

    }
    [StructLayout(LayoutKind.Sequential, Size = 52)]

    public struct Ray(Vector3 origin, Vector3 direction, Vector3 viewDir, float radius, float near, float far, float lossMult)
    {
        public Vector3 Origin = origin;
        public Vector3 Direction = direction;
        public Vector3 ViewDir = viewDir;
        public float Radius = radius;
        public float Near = near;
        public float Far = far;
        public float LossMult = lossMult;
    }

}
