using System.Numerics;

namespace ScratchNerf
{
    public class MLP
    {
        // Fields and Properties
        public int NetDepth { get; set; } = 8;          // The depth of the first part of MLP.
        public int NetWidth { get; set; } = 256;        // The width of the first part of MLP.
        public int NetDepthCondition { get; set; } = 1; // The depth of the second part of MLP.
        public int NetWidthCondition { get; set; } = 128; // The width of the second part of MLP.
        public Func<float, float> NetActivation { get; set; } = ReLU; // Activation function (ReLU).
        public Func<float, float> NetActivationGrad { get; set; } = ReLUGrad; // Gradient of the activation function.
        public int NumRgbChannels { get; set; } = 3;    // Number of RGB channels.
        public int NumDensityChannels { get; set; } = 1; // Number of density channels.
        public int LocationDimension { get; set; } = 3; // Location dimension.
        public int DirectionDimension { get; set; } = 3; // Direction dimension.
        public int LocationEncodings { get; set; } = 1;
        public int DirectionEncodings { get; set; } = 1;
        public int SkipLayer { get; set; } = 4;         // Skip connection every N layers.
        public Random rand = new();
        private float[][,] weights;
        private float[][] biases;
        private float[][] inputs;
        //flatten weights and biases into 1Darray
        public float[] allParams
        {
            get => weights.SelectMany(arr => arr.Cast<float>())
                .Concat(biases.SelectMany(arr => arr)).ToArray();
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
            float[] x = new float[LocationDimension * LocationEncodings];
            for (int i = 0; i < LocationEncodings; i++)
                for (int j = 0; j < LocationDimension; j++)
                    x[i * LocationDimension + j] = positionEncoded[i][j];
            float[] inputs = (float[])x.Clone();
            for (int i = 0; i < NetDepth; i++)
            {
                this.inputs[i] = x;
                if (i % SkipLayer == 0 && i > 0) x = [.. x, .. inputs];
                x = ApplyLayer(x, weights[i], biases[i], NetActivation);
            }
            this.inputs[NetDepth] = x;
            float rawDensity = ApplyLayer(x, weights[NetDepth], biases[NetDepth], f => f)[0];
            float[] direction = new float[DirectionDimension * DirectionEncodings];
            for (int i = 0; i < DirectionEncodings; i++)
                for (int j = 0; j < DirectionDimension; j++)
                    direction[i * DirectionDimension + j] = directionEncoded[i][j];
            x = [.. x, .. direction];
            for (int i = 0; i < NetDepthCondition; i++)
            {
                this.inputs[NetDepth + 1 + i] = x;
                x = ApplyLayer(x, weights[NetDepth + 1 + i], biases[NetDepth + 1 + i], NetActivation);
            }
            float[] rawRgb = ApplyLayer(x, weights[NetDepth + NetDepthCondition + 1], biases[NetDepth + NetDepthCondition + 1], f => f);
            Vector3 rawRgbVec = new(rawRgb[0], rawRgb[1], rawRgb[2]);
            return (rawRgbVec, rawDensity);
        }
        public (Vector3 RawRgb, float RawDensity, float[][] inputs) CallCached(Vector3[] positionEncoded, Vector3[] directionEncoded)
        {
            float[] x = new float[LocationDimension * LocationEncodings];
            for (int i = 0; i < LocationEncodings; i++)
            for (int j = 0; j < LocationDimension; j++)
                x[i * LocationDimension + j] = positionEncoded[i][j];
            float[] inputs = (float[])x.Clone();
            for (int i = 0; i < NetDepth; i++)
            {
                if (i % SkipLayer == 0 && i > 0) x = [.. x, .. inputs];
                this.inputs[i] = x;
                x = ApplyLayer(x, weights[i], biases[i], NetActivation);
            }
            this.inputs[NetDepth] = x;
            float rawDensity = ApplyLayer(x, weights[NetDepth], biases[NetDepth], f => f)[0];
            float[] direction = new float[DirectionDimension * DirectionEncodings];
            for (int i = 0; i < DirectionEncodings; i++)
            for (int j = 0; j < DirectionDimension; j++)
                direction[i * DirectionDimension + j] = directionEncoded[i][j];
            x = [.. x, .. direction];
            for (int i = 0; i < NetDepthCondition; i++)
            {
                this.inputs[NetDepth + 1 + i] = x;
                x = ApplyLayer(x, weights[NetDepth + 1 + i], biases[NetDepth + 1 + i], NetActivation);
            }
            float[] rawRgb = ApplyLayer(x, weights[^1], biases[^1], f => f);
            Vector3 rawRgbVec = new(rawRgb[0], rawRgb[1], rawRgb[2]);
            return (rawRgbVec, rawDensity, this.inputs);
        }

        public float[] GetGradient(float[][] inputs, Vector3 rawRgbGradient, float rawDensityGradient)
        {
            float[] rawRgbArrGrad = [rawRgbGradient.X, rawRgbGradient.Y, rawRgbGradient.Z];
            float[][,] weightGrads = new float[NetDepth + NetDepthCondition + 2][,];
            float[][] biasGrads = new float[NetDepth + NetDepthCondition + 2][];
            (float[] xGrad, weightGrads[^1], biasGrads[^1]) = GetLayerGradient(inputs[^1], weights[^1], biases[^1], rawRgbArrGrad, f => 1);
            for(int i = NetDepthCondition - 1; i >= 0; i--)
            {
                (xGrad, weightGrads[NetDepth + 1 + i], biasGrads[NetDepth + 1 + i]) = GetLayerGradient(inputs[NetDepth + 1 + i], weights[NetDepth + 1 + i], biases[NetDepth + 1 + i], xGrad, NetActivationGrad);
            }
            xGrad = xGrad.Take(xGrad.Length - DirectionDimension * DirectionEncodings).ToArray();
            (float[] xGradFromDensityToCombine, weightGrads[NetDepth], biasGrads[NetDepth]) = GetLayerGradient(inputs[NetDepth], weights[NetDepth], biases[NetDepth], new float[] { rawDensityGradient }, f => 1);
            for(int i = 0; i < xGrad.Length; i++)
            {
                xGrad[i] += xGradFromDensityToCombine[i];
            }
            for(int i = NetDepth - 1; i >= 0; i--)
            {
                (xGrad, weightGrads[i], biasGrads[i]) = GetLayerGradient(inputs[i], weights[i], biases[i], xGrad, NetActivationGrad);
            }
            return weightGrads.SelectMany(arr => arr.Cast<float>())
                .Concat(biasGrads.SelectMany(arr => arr)).ToArray();
        }

        private static float[] ApplyLayer(float[] inputs, float[,] weights, float[] biases, Func<float, float> activation)
        {
            if (weights.GetLength(1) != inputs.Length)
                throw new ArgumentException("Weights and inputs must have the same number of features.");
            if (biases.Length != weights.GetLength(0))
                throw new ArgumentException("Biases must have the same number of neurons as the weights.");
            int numSamples = inputs.Length;
            int outputDim = weights.GetLength(0);
            float[] outputs = new float[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                float sum = 0f;
                for (int j = 0; j < outputDim; j++) sum += inputs[i] * weights[j, i];
                outputs[i] = activation(sum + biases[i]);
            }

            return outputs;
        }
        private static (float[] inputGradient, float[,] weightGradient, float[] biasGradient) GetLayerGradient(float[] inputs, float[,] weights, float[] biases, float[] outputGradient, Func<float, float> activationGradient)
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
                float sum = 0f;
                for (int i = 0; i < inputDim; i++)
                    sum += inputs[i] * weights[j, i];
                activationGrads[j] = activationGradient(sum + biases[j]);
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
