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

        Random rng = new();
        public MLP mlp;

        public MipNerfModel() => mlp = new(MaxDegPoint - MinDegPoint, DegView);

        // Methods
        public (Vector3 CompositeRgb, float Distance, float Accumulation)[,] Call(Ray[] rays, bool randomized, bool whiteBackground)
        {
            MLP mlp = new(MaxDegPoint - MinDegPoint, DegView);
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
                Vector3[,][] samplesEnc = new Vector3[][numRays, NumSamples + 1 - iLevel];
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

        public static (MipNerfModel, (Vector3 CompositeRgb, float Distance, float Accumulation)[,]) ConstructMipNerf(Ray[] rays)
        {
            MipNerfModel model = new();
            (Vector3 CompositeRgb, float Distance, float Accumulation)[,] results = model.Call(rays, false, false);
            return (model, results);
        }

    }

    public struct Ray(Vector3 origin, Vector3 direction, float radius, float near, float far, float lossMult)
    {
        public Vector3 Origin = origin;
        public Vector3 Direction = direction;
        public float Radius = radius;
        public float Near = near;
        public float Far = far;
        public float LossMult = lossMult;
    }

}
