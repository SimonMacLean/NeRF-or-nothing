using System.Numerics;

namespace ScratchNerf
{
    public class MLP
    {
        // Fields and Properties
        public int NetDepth  = 8;          // The depth of the first part of MLP.
        public int NetWidth  = 256;        // The width of the first part of MLP.
        public int NetDepthCondition  = 1; // The depth of the second part of MLP.
        public int NetWidthCondition  = 128; // The width of the second part of MLP.
        public Func<float, float> NetActivation  = ReLU; // Activation function (ReLU).
        public Func<float, float> NetActivationGrad  = ReLUGrad; // Gradient of the activation function.
        public int NumRgbChannels  = 3;    // Number of RGB channels.
        public int NumDensityChannels  = 1; // Number of density channels.
        public int LocationDimension  = 3; // Location dimension.
        public int DirectionDimension  = 3; // Direction dimension.
        public int LocationEncodings  = 1;
        public int DirectionEncodings  = 1;
        public int SkipLayer  = 4;         // Skip connection every N layers.
        public Random rand = new();
        private float[][,] weights;
        private float[][] biases;
        private float[][] inputs;
        private float[][] weightedSums;

        public float[] allParams
        {
            get
            {
                int numParams = weights.Sum(arr => arr.Length) + biases.Sum(arr => arr.Length);
                float[] allParams = new float[numParams];
                int numIterated = 0;
                foreach (float[,] weight in weights)
                {
                    Buffer.BlockCopy(weight, 0, allParams, numIterated * sizeof(float), weight.Length * sizeof(float));
                    numIterated += weight.Length;
                }
                foreach (float[] bias in biases)
                {
                    Buffer.BlockCopy(bias, 0, allParams, numIterated * sizeof(float), bias.Length * sizeof(float));
                    numIterated += bias.Length;
                }
                return allParams;
            }
            set
            {
                int numIterated = 0;
                foreach (float[,] weight in weights)
                    for (int j = 0; j < weight.GetLength(0); j++)
                    for (int k = 0; k < weight.GetLength(1); k++)
                    {
                        weight[j, k] = value[numIterated];
                        numIterated++;
                    }
                foreach (float[] bias in biases)
                    for (int j = 0; j < bias.Length; j++)
                    {
                        bias[j] = value[numIterated];
                        numIterated++;
                    }
            }
        }
        public MLP(int degPoint, int degView)
        {
            LocationEncodings = 2 * degPoint;
            DirectionEncodings = 2 * degView + 1;
            weights = new float[NetDepth + NetDepthCondition + 2][,];
            biases = new float[NetDepth + NetDepthCondition + 2][];
            inputs = new float[NetDepth + NetDepthCondition + 2][];
            weightedSums = new float[NetDepth + NetDepthCondition + 2][];
            weights[0] = new float[NetWidth, LocationDimension * LocationEncodings];
            for (int i = 1; i < NetDepth; i++) weights[i] = new float[NetWidth, i % SkipLayer == 0 ? NetWidth + LocationDimension * LocationEncodings : NetWidth];
            weights[NetDepth] = new float[NumDensityChannels, NetWidth];
            weights[NetDepth + 1] = new float[NetWidthCondition, NetWidth + DirectionDimension * DirectionEncodings];
            for (int i = 1; i < NetDepthCondition; i++) weights[NetDepth + 1 + i] = new float[NetWidthCondition, NetWidthCondition];
            weights[NetDepth + 1 + NetDepthCondition] = new float[NumRgbChannels, NetWidthCondition];
            for (int i = 0; i < NetDepth + NetDepthCondition + 2; i++) biases[i] = new float[weights[i].GetLength(0)];
            foreach (float[,] t in weights)
            {
                int fanIn = t.GetLength(1);
                int fanOut = t.GetLength(0);
                for (int j = 0; j < fanOut; j++)
                    for (int k = 0; k < fanIn; k++) t[j, k] = rand.GlorotUniform(fanIn, fanOut);
            }
        }
        public (Vector3 RawRgb, float RawDensity) Call(Vector3[] positionEncoded, Vector3[] directionEncoded)
        {
            inputs[0] = new float[LocationDimension * LocationEncodings];
            for (int i = 0; i < LocationEncodings; i++)
            for (int j = 0; j < LocationDimension; j++)
                inputs[0][i * LocationDimension + j] = positionEncoded[i][j];
            for (int i = 0; i < NetDepth; i++)
            {
                if (i % SkipLayer == 0 && i > 0) inputs[i] = [.. inputs[i], .. inputs[0]];
                (inputs[i + 1], weightedSums[i]) = ApplyLayer(inputs[i], weights[i], biases[i], NetActivation);
            }
            (float[] rawDensity, weightedSums[NetDepth]) = ApplyLayer(inputs[NetDepth], weights[NetDepth], biases[NetDepth], f => f);
            float[] direction = new float[DirectionDimension * DirectionEncodings];
            for (int i = 0; i < DirectionEncodings; i++)
            for (int j = 0; j < DirectionDimension; j++)
                direction[i * DirectionDimension + j] = directionEncoded[i][j];
            inputs[NetDepth + 1] = [.. inputs[NetDepth], .. direction];
            for (int i = 0; i < NetDepthCondition; i++)
            {
                (inputs[NetDepth + 2 + i], weightedSums[NetDepth + 1 + i]) = ApplyLayer(inputs[NetDepth + 1 + i], weights[NetDepth + 1 + i], biases[NetDepth + 1 + i], NetActivation);
            }
            (float[] rawRgb, weightedSums[NetDepth + 1 + NetDepthCondition]) = ApplyLayer(inputs[NetDepth + 1 + NetDepthCondition], weights[^1], biases[^1], f => f);
            Vector3 rawRgbVec = new(rawRgb[0], rawRgb[1], rawRgb[2]);
            return (rawRgbVec, rawDensity[0]);
        }
        public (Vector3 RawRgb, float RawDensity, float[][] inputs) CallCached(Vector3[] positionEncoded, Vector3[] directionEncoded)
        {
            inputs[0] = new float[LocationDimension * LocationEncodings];
            for (int i = 0; i < LocationEncodings; i++)
            for (int j = 0; j < LocationDimension; j++)
                inputs[0][i * LocationDimension + j] = positionEncoded[i][j];
            for (int i = 0; i < NetDepth; i++)
            {
                if (i % SkipLayer == 0 && i > 0) inputs[i] = [.. inputs[i], .. inputs[0]];
                (inputs[i + 1], weightedSums[i]) = ApplyLayer(inputs[i], weights[i], biases[i], NetActivation);
            }
            (float[] rawDensity, weightedSums[NetDepth]) = ApplyLayer(inputs[NetDepth], weights[NetDepth], biases[NetDepth], f => f);
            float[] direction = new float[DirectionDimension * DirectionEncodings];
            for (int i = 0; i < DirectionEncodings; i++)
            for (int j = 0; j < DirectionDimension; j++)
                direction[i * DirectionDimension + j] = directionEncoded[i][j];
            inputs[NetDepth + 1] = [.. inputs[NetDepth], .. direction];
            for (int i = 0; i < NetDepthCondition; i++)
            {
                (inputs[NetDepth + 2 + i], weightedSums[NetDepth + 1 + i]) = ApplyLayer(inputs[NetDepth + 1 + i], weights[NetDepth + 1 + i], biases[NetDepth + 1 + i], NetActivation);
            }
            (float[] rawRgb, weightedSums[NetDepth + 1 + NetDepthCondition]) = ApplyLayer(inputs[NetDepth + 1 + NetDepthCondition], weights[^1], biases[^1], f => f);
            Vector3 rawRgbVec = new(rawRgb[0], rawRgb[1], rawRgb[2]);
            return (rawRgbVec, rawDensity[0], (float[][])inputs.Clone());
        }

        public float[] GetGradient(float[][] inputs, Vector3 rawRgbGradient, float rawDensityGradient)
        {
            float[] rawRgbArrGrad = [rawRgbGradient.X, rawRgbGradient.Y, rawRgbGradient.Z];
            float[][,] weightGrads = new float[NetDepth + NetDepthCondition + 2][,];
            float[][] biasGrads = new float[NetDepth + NetDepthCondition + 2][];
            (float[] xGrad, weightGrads[^1], biasGrads[^1]) = GetLayerGradient(inputs[^1], weights[^1], biases[^1], weightedSums[^1], rawRgbArrGrad, f => 1);
            for(int i = NetDepthCondition - 1; i >= 0; i--)
            {
                (xGrad, weightGrads[NetDepth + 1 + i], biasGrads[NetDepth + 1 + i]) = GetLayerGradient(inputs[NetDepth + 1 + i], weights[NetDepth + 1 + i], biases[NetDepth + 1 + i], weightedSums[NetDepth + 1 + i], xGrad, NetActivationGrad);
            }
            xGrad = xGrad.Take(xGrad.Length - DirectionDimension * DirectionEncodings).ToArray();
            (float[] xGradFromDensityToCombine, weightGrads[NetDepth], biasGrads[NetDepth]) = GetLayerGradient(inputs[NetDepth], weights[NetDepth], biases[NetDepth], weightedSums[NetDepth], new float[] { rawDensityGradient }, f => 1);
            for(int i = 0; i < xGrad.Length; i++)
            {
                xGrad[i] += xGradFromDensityToCombine[i];
            }

            for (int i = NetDepth - 1; i >= 0; i--)
            {
                (xGrad, weightGrads[i], biasGrads[i]) =
                    GetLayerGradient(inputs[i], weights[i], biases[i], weightedSums[i], xGrad, NetActivationGrad);
            }
            int numGrads = weightGrads.Sum(arr => arr.Length) + biasGrads.Sum(arr => arr.Length);
            float[] grads = new float[numGrads];
            int numIterated = 0;
            foreach (float[,] weightGrad in weightGrads)
            {
                Buffer.BlockCopy(weightGrad, 0, grads, numIterated * sizeof(float), weightGrad.Length * sizeof(float));
                numIterated += weightGrad.Length;
            }
            foreach (float[] biasGrad in biasGrads)
            {
                Buffer.BlockCopy(biasGrad, 0, grads, numIterated * sizeof(float), biasGrad.Length * sizeof(float));
                numIterated += biasGrad.Length;
            }

            return grads;
        }

        private static (float[] outputs, float[] weightedSums) ApplyLayer(float[] inputs, float[,] weights, float[] biases, Func<float, float> activation)
        {
            if (weights.GetLength(1) != inputs.Length)
                throw new ArgumentException("Weights and inputs must have the same number of features.");
            if (biases.Length != weights.GetLength(0))
                throw new ArgumentException("Biases must have the same number of neurons as the weights.");
            int inputDim = inputs.Length;
            int outputDim = weights.GetLength(0);
            float[] outputs = new float[outputDim];
            float[] weightedSums = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++) weightedSums[i] += inputs[j] * weights[i, j];
                weightedSums[i] += biases[i];
                outputs[i] = activation(weightedSums[i]);
            }

            return (outputs, weightedSums);
        }
        private static (float[] inputGradient, float[,] weightGradient, float[] biasGradient) GetLayerGradient(float[] inputs, float[,] weights, float[] biases, float[] weightedSums, float[] outputGradient, Func<float, float> activationGradient)
        {
            int inputDim = inputs.Length;
            int outputDim = weights.GetLength(0);

            float[] inputGradient = new float[inputDim];
            float[,] weightGradient = new float[outputDim, inputDim];
            float[] biasGradient = new float[outputDim];

            // Calculate the gradient of the activation function
            float[] activationGrads = new float[outputDim];
            for (int j = 0; j < outputDim; j++)
            {
                activationGrads[j] = activationGradient(weightedSums[j] + biases[j]);
                float grad = outputGradient[j] * activationGrads[j];
                biasGradient[j] = grad;
                for (int i = 0; i < inputDim; i++)
                {
                    weightGradient[j, i] = grad * inputs[i];
                    inputGradient[i] += grad * weights[j, i];
                }
            }

            return (inputGradient, weightGradient, biasGradient);
        }

        // Helper function: ReLU activation
        private static float ReLU(float x) => Math.Max(0, x);
        private static float ReLUGrad(float x) => x > 0 ? 1 : 0;
    }
}
