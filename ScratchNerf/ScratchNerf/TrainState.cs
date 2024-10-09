using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ScratchNerf
{
    public class TrainState(Optimizer optimizer)
    {
        public Optimizer Optimizer { get; set; } = optimizer;
    }
    public abstract class Optimizer
    {
        public abstract float[] Step(float[] variables, float[] gradients);
    }
    public class AdamOptimizer(int paramCount, float learningRate) : Optimizer {
        public float LearningRate { get; set; } = learningRate;
        public float Beta1 { get; set; } = 0.9f;
        public float Beta2 { get; set; } = 0.999f;
        public float Epsilon { get; set; } = 1e-8f;
        public int Iteration { get; set; } = 0;
        public float[] M { get; set; } = new float[paramCount];
        public float[] V { get; set; } = new float[paramCount];
        public override float[] Step(float[] variables, float[] gradients)
        {
            Iteration++;
            for (int i = 0; i < variables.Length; i++)
            {
                M[i] = Beta1 * M[i] + (1 - Beta1) * gradients[i];
                V[i] = Beta2 * V[i] + (1 - Beta2) * gradients[i] * gradients[i];
                float mHat = M[i] / (1 - MathF.Pow(Beta1, Iteration));
                float vHat = V[i] / (1 - MathF.Pow(Beta2, Iteration));
                variables[i] -= LearningRate * mHat / (MathF.Sqrt(vHat) + Epsilon);
            }
            return variables;
        }
    }
    public enum DatasetType
    {
        Multicam,
        Blender,
        LLFF
    }
    public static class Config
    {

        public static DatasetType DatasetLoader { get; set; } = DatasetType.LLFF;  // The type of dataset loader to use.
        public static int BatchSize { get; set; } = 1024;  // The number of rays/pixels in each batch.
        public static int Factor { get; set; } = 0;  // The downsample factor of images, 0 for no downsampling.
        public static bool Spherify { get; set; } = false;  // Set to True for spherical 360 scenes.
        public static bool RenderPath { get; set; } = false;  // If True, render a path. Used only by LLFF.
        public static int LlffHold { get; set; } = 8;  // Use every Nth image for the test set. Used only by LLFF.
        public static float LrInit { get; set; } = 5e-4f;  // The initial learning rate.
        public static float LrFinal { get; set; } = 5e-6f;  // The final learning rate.
        public static int LrDelaySteps { get; set; } = 2500;  // The number of "warmup" learning steps.
        public static float LrDelayMult { get; set; } = 0.01f;  // How much sever the "warmup" should be.
        public static float GradMaxNorm { get; set; } = 0f;  // Gradient clipping magnitude, disabled if == 0.
        public static float GradMaxVal { get; set; } = 0f;  // Gradient clipping value, disabled if == 0.
        public static int MaxSteps { get; set; } = 1000000;  // The number of optimization steps.
        public static int SaveEvery { get; set; } = 100000;  // The number of steps to save a checkpoint.
        public static int PrintEvery { get; set; } = 100;  // The number of steps to print the loss.
        public static int GcEvery { get; set; } = 10000;  // The number of steps to render an image.
        public static int TestRenderInterval { get; set; } = 1;  // The interval between images saved to disk.
        public static bool DisableMultiscaleLoss { get; set; } = false;  // If True, disable multiscale loss.
        public static bool Randomized { get; set; } = true;  // Use randomized stratified sampling.
        public static float Near { get; set; } = 2f;  // Near plane distance.
        public static float Far { get; set; } = 6f;  // Far plane distance.
        public static float CoarseLossMult { get; set; } = 0.1f;  // How much to downweight the coarse loss(es).
        public static float WeightDecayMult { get; set; } = 0f;  // The multiplier on weight decay.
        public static bool WhiteBkgd { get; set; } = true;  // If True, use white as the background (black o.w.).
    }
}