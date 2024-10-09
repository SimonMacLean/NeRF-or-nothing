using System.Numerics;
using System.Runtime.InteropServices;

namespace ScratchNerf
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Size = 36)]
    // ReSharper disable once InconsistentNaming
    public struct Matrix3x3(
        float m11 = 0,
        float m12 = 0,
        float m13 = 0,
        float m21 = 0,
        float m22 = 0,
        float m23 = 0,
        float m31 = 0,
        float m32 = 0,
        float m33 = 0)
    {
        private float _m11 = m11, _m12 = m12, _m13 = m13, _m21 = m21, _m22 = m22, _m23 = m23, _m31 = m31, _m32 = m32, _m33 = m33;
        public static readonly Matrix3x3 Identity = new(1, 0, 0, 0, 1, 0, 0, 0, 1);
        public static readonly Matrix3x3 Zero = new(0, 0, 0, 0, 0, 0, 0, 0, 0);
        public readonly Matrix3x3 transpose => Transpose(this);
        public static Matrix3x3 operator *(Matrix3x3 a, Matrix3x3 b) =>
            new(
                a._m11 * b._m11 + a._m12 * b._m21 + a._m13 * b._m31,
                a._m11 * b._m12 + a._m12 * b._m22 + a._m13 * b._m32,
                a._m11 * b._m13 + a._m12 * b._m23 + a._m13 * b._m33,
                a._m21 * b._m11 + a._m22 * b._m21 + a._m23 * b._m31,
                a._m21 * b._m12 + a._m22 * b._m22 + a._m23 * b._m32,
                a._m21 * b._m13 + a._m22 * b._m23 + a._m23 * b._m33,
                a._m31 * b._m11 + a._m32 * b._m21 + a._m33 * b._m31,
                a._m31 * b._m12 + a._m32 * b._m22 + a._m33 * b._m32,
                a._m31 * b._m13 + a._m32 * b._m23 + a._m33 * b._m33);

        public static Vector3 operator *(Matrix3x3 a, Vector3 b) =>
            new(
                a._m11 * b.X + a._m12 * b.Y + a._m13 * b.Z,
                a._m21 * b.X + a._m22 * b.Y + a._m23 * b.Z,
                a._m31 * b.X + a._m32 * b.Y + a._m33 * b.Z);

        public static Matrix3x3 Transpose(Matrix3x3 a) => new(a._m11, a._m21, a._m31, a._m12, a._m22, a._m32, a._m13, a._m23, a._m33);
        public static Matrix3x3 operator +(Matrix3x3 a, Matrix3x3 b) => new(a._m11 + b._m11, a._m12 + b._m12, a._m13 + b._m13, a._m21 + b._m21, a._m22 + b._m22, a._m23 + b._m23, a._m31 + b._m31, a._m32 + b._m32, a._m33 + b._m33);
        public static Matrix3x3 operator *(Matrix3x3 a, float b) => new(a._m11 * b, a._m12 * b, a._m13 * b, a._m21 * b, a._m22 * b, a._m23 * b, a._m31 * b, a._m32 * b, a._m33 * b);
        public static Matrix3x3 operator /(Matrix3x3 a, float b) => new(a._m11 / b, a._m12 / b, a._m13 / b, a._m21 / b, a._m22 / b, a._m23 / b, a._m31 / b, a._m32 / b, a._m33 / b);
        public static Matrix3x3 operator -(Matrix3x3 a, Matrix3x3 b) => new(a._m11 - b._m11, a._m12 - b._m12, a._m13 - b._m13, a._m21 - b._m21, a._m22 - b._m22, a._m23 - b._m23, a._m31 - b._m31, a._m32 - b._m32, a._m33 - b._m33);
        public static Matrix3x3 operator -(Matrix3x3 a) => new(-a._m11, -a._m12, -a._m13, -a._m21, -a._m22, -a._m23, -a._m31, -a._m32, -a._m33);
        public float this[int i, int j]
        {
            get =>
                (i, j) switch
                {
                    (0, 0) => _m11,
                    (0, 1) => _m12,
                    (0, 2) => _m13,
                    (1, 0) => _m21,
                    (1, 1) => _m22,
                    (1, 2) => _m23,
                    (2, 0) => _m31,
                    (2, 1) => _m32,
                    (2, 2) => _m33,
                    _ => throw new IndexOutOfRangeException()
                };
            set
            {
                switch (i, j)
                {
                    case (0, 0):
                        _m11 = value;
                        break;
                    case (0, 1):
                        _m12 = value;
                        break;
                    case (0, 2):
                        _m13 = value;
                        break;
                    case (1, 0):
                        _m21 = value;
                        break;
                    case (1, 1):
                        _m22 = value;
                        break;
                    case (1, 2):
                        _m23 = value;
                        break;
                    case (2, 0):
                        _m31 = value;
                        break;
                    case (2, 1):
                        _m32 = value;
                        break;
                    case (2, 2):
                        _m33 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public Vector3 this[int i]
        {
            get => i switch
            {
                0 => new Vector3(_m11, _m12, _m13),
                1 => new Vector3(_m21, _m22, _m23),
                2 => new Vector3(_m31, _m32, _m33),
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (i)
                {
                    case 0:
                        _m11 = value.X;
                        _m12 = value.Y;
                        _m13 = value.Z;
                        break;
                    case 1:
                        _m21 = value.X;
                        _m22 = value.Y;
                        _m23 = value.Z;
                        break;
                    case 2:
                        _m31 = value.X;
                        _m32 = value.Y;
                        _m33 = value.Z;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public Vector3 Diagonal
        {
            get => new(_m11, _m22, _m33);
            set
            {
                _m11 = value.X;
                _m22 = value.Y;
                _m33 = value.Z;
            }
        }
    }
    public struct MatrixNxN(int dimension)
    {
        public int Dimension = dimension;
        float[,] Values = new float[dimension, dimension];
        private MatrixNxN transpose => Transpose(this);
        public float this[int i, int j]
        {
            get => Values[i, j];
            set => Values[i, j] = value;
        }
        public static MatrixNxN Identity(int dimension)
        {
            MatrixNxN result = new(dimension);
            for (int i = 0; i < dimension; i++) result[i, i] = 1;
            return result;
        }
        public static MatrixNxN operator *(MatrixNxN a, MatrixNxN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Matrices must have the same dimension for multiplication.");
            int dimension = a.Dimension;
            MatrixNxN result = new(dimension);
            for (int i = 0; i < dimension; i++)
            for (int j = 0; j < dimension; j++)
            {
                float sum = 0;
                for (int k = 0; k < dimension; k++) sum += a[i, k] * b[k, j];
                result[i, j] = sum;
            }
            return result;
        }
        public static MatrixNxN operator +(MatrixNxN a, MatrixNxN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Matrices must have the same dimension for addition.");
            int dimension = a.Dimension;
            MatrixNxN result = new(dimension);
            for (int i = 0; i < dimension; i++)
            for (int j = 0; j < dimension; j++)
                result[i, j] = a[i, j] + b[i, j];
            return result;
        }
        public static MatrixNxN operator -(MatrixNxN a, MatrixNxN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Matrices must have the same dimension for subtraction.");
            int dimension = a.Dimension;
            MatrixNxN result = new(dimension);
            for (int i = 0; i < dimension; i++)
            for (int j = 0; j < dimension; j++)
                result[i, j] = a[i, j] - b[i, j];
            return result;
        }
        public static MatrixNxN operator *(MatrixNxN a, float b)
        {
            int dimension = a.Dimension;
            MatrixNxN result = new(dimension);
            for (int i = 0; i < dimension; i++)
            for (int j = 0; j < dimension; j++)
                result[i, j] = a[i, j] * b;
            return result;
        }
        public static MatrixNxN operator /(MatrixNxN a, float b) => a * (1 / b);
        public static MatrixNxN operator -(MatrixNxN a) => a * -1;
        public static MatrixNxN Transpose(MatrixNxN a)
        {
            int dimension = a.Dimension;
            MatrixNxN result = new(dimension);
            for (int i = 0; i < dimension; i++)
            for (int j = 0; j < dimension; j++)
                result[i, j] = a[j, i];
            return result;
        }
        public static VectorN operator *(MatrixNxN a, VectorN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Matrix and vector must have the same dimension for multiplication.");
            int dimension = a.Dimension;
            float[] result = new float[dimension];
            for (int i = 0; i < dimension; i++)
            {
                float sum = 0;
                for (int j = 0; j < dimension; j++)
                    sum += a[i, j] * b[j];
                result[i] = sum;
            }
            return new VectorN(result);
        }
        public VectorN this[int i]
        {
            get
            {
                float[] row = new float[Dimension];
                for (int j = 0; j < Dimension; j++)
                    row[j] = Values[i, j];
                return new VectorN(row);
            }
            set
            {
                for (int j = 0; j < Dimension; j++)
                    Values[i, j] = value[j];
            }
        }

    }
    public struct VectorN(float[] coordinates)
    {
        public int Dimension = coordinates.Length;

        public float this[int i]
        {
            get => coordinates[i];
            set => coordinates[i] = value;
        }
        public static VectorN Zero(int dimension) => new(new float[dimension]);

        public static VectorN One(int dimension) => new(Enumerable.Repeat(1f, dimension).ToArray());
        public static VectorN operator +(VectorN a, VectorN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Vectors must have the same dimension for addition.");
            int dimension = a.Dimension;
            float[] result = new float[dimension];
            for (int i = 0; i < dimension; i++)
                result[i] = a[i] + b[i];
            return new VectorN(result);
        }
        public static VectorN operator -(VectorN a, VectorN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Vectors must have the same dimension for subtraction.");
            int dimension = a.Dimension;
            float[] result = new float[dimension];
            for (int i = 0; i < dimension; i++)
                result[i] = a[i] - b[i];
            return new VectorN(result);
        }
        public static VectorN operator *(VectorN a, float b)
        {
            int dimension = a.Dimension;
            float[] result = new float[dimension];
            for (int i = 0; i < dimension; i++)
                result[i] = a[i] * b;
            return new VectorN(result);
        }
        public static VectorN operator /(VectorN a, float b)
        {
            int dimension = a.Dimension;
            float[] result = new float[dimension];
            for (int i = 0; i < dimension; i++)
                result[i] = a[i] / b;
            return new VectorN(result);
        }
        public static VectorN operator -(VectorN a)
        {
            int dimension = a.Dimension;
            float[] result = new float[dimension];
            for (int i = 0; i < dimension; i++)
                result[i] = -a[i];
            return new VectorN(result);
        }
        public static MatrixNxN Outer(VectorN a, VectorN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Vectors must have the same dimension for outer product.");
            int dimension = a.Dimension;
            MatrixNxN result = new(dimension);
            for (int i = 0; i < dimension; i++)
            for (int j = 0; j < dimension; j++)
                result[i, j] = a[i] * b[j];
            return result;
        }
        public static float Dot(VectorN a, VectorN b)
        {
            if (a.Dimension != b.Dimension)
                throw new ArgumentException("Vectors must have the same dimension for dot product.");
            int dimension = a.Dimension;
            float sum = 0;
            for (int i = 0; i < dimension; i++)
                sum += a[i] * b[i];
            return sum;
        }

    }
    // ReSharper disable once InconsistentNaming
    public static class MipHelpers
    {
        public enum RayShape {
            Conical,
            Cylindrical
        }
        public static Vector3[] PositionalEncoding(Vector3 x, int minDeg, int maxDeg)
        {
            int numScales = maxDeg - minDeg;
            float[] scales = Enumerable.Range(minDeg, numScales)
                .Select(i => (float)(1<<i))
                .ToArray();
            int resultNumCols = numScales * 2 + 1;
            Vector3[] result = new Vector3[numScales * 2 + 1];
            result[0] = x;
            for (int j = 0; j < numScales; j++)
            {
                float scale = scales[j];
                Vector3 xb = x * scale;
                Vector3 sinValue = new(MathF.Sin(xb.X), MathF.Sin(xb.Y), MathF.Sin(xb.Z));
                Vector3 cosValue = new(MathF.Cos(xb.X), MathF.Cos(xb.Y), MathF.Cos(xb.Z));
                result[j * 2 + 1] = sinValue;
                result[j * 2 + 2] = cosValue;
            }
            return result;
        }

        public static (float mean, float variance) ExpectedSin(float x, float xVar)
        {
            float expNegHalfXVar = MathF.Exp(-0.5f * xVar);
            float y = expNegHalfXVar * MathF.Sin(x);
            float expNeg2XVar = MathF.Exp(-2f * xVar);
            float cos2x = MathF.Cos(2f * x);
            float yVar = MathF.Max(0, 0.5f * (1 - expNeg2XVar * cos2x) - y * y);
            return (y, yVar);
        }
        public static (Vector3 mean, Matrix3x3 covariance) LiftGaussian(Vector3 direction, float tMean, float tVar, float rVar, bool diagonal)
        {
            Vector3 mean = direction * tMean;
            float directionMagnitudeSquared = MathF.Max(1e-10f, Vector3.Dot(direction, direction));
            Matrix3x3 covariance = new();
            if (diagonal)
            {
                Vector3 dOuterDiagonal = new (direction.X * direction.X, direction.Y * direction.Y, direction.Z * direction.Z);
                Vector3 nullOuterDiagonal = Vector3.One - dOuterDiagonal / directionMagnitudeSquared;
                Vector3 tCovarianceDiagonal = tVar * dOuterDiagonal;
                Vector3 xyCovarianceDiagonal = rVar * nullOuterDiagonal;
                covariance.Diagonal = tCovarianceDiagonal + xyCovarianceDiagonal;
            }
            else
            {
                Matrix3x3 dOuter = new(
                    direction.X * direction.X, direction.X * direction.Y, direction.X * direction.Z,
                    direction.Y * direction.X, direction.Y * direction.Y, direction.Y * direction.Z,
                    direction.Z * direction.X, direction.Z * direction.Y, direction.Z * direction.Z);
                Matrix3x3 nullOuter = Matrix3x3.Identity - dOuter / directionMagnitudeSquared;
                covariance = dOuter * tVar + nullOuter * rVar;
            }
            return (mean, covariance);
        }
        public static (Vector3 mean, Matrix3x3 covariance) ConicalFrustumToGaussian(Vector3 direction, float startDistance, float endDistance, float baseRadius, bool diagonalCovariance)
        {
            float meanDistance = (startDistance + endDistance) / 2;
            float halfWidth = (endDistance - startDistance) / 2;
            float meanDistanceSquared = meanDistance * meanDistance;
            float halfWidthSquared = halfWidth * halfWidth;
            float denominator = 3 * meanDistanceSquared + halfWidthSquared;
            float distanceMean = meanDistance + (2 * meanDistance * halfWidthSquared) / denominator;
            float distanceVariance = halfWidthSquared / 3 - (4f / 15f) * (halfWidthSquared * halfWidthSquared * (12 * meanDistanceSquared - halfWidthSquared)) / (denominator * denominator);
            float radiusVariance = baseRadius * baseRadius * (meanDistanceSquared / 4 + (5f / 12f) * halfWidthSquared - (4f / 15f) * (halfWidthSquared * halfWidthSquared) / denominator);
            return LiftGaussian(direction, distanceMean, distanceVariance, radiusVariance, diagonalCovariance);
        }
        public static (Vector3 mean, Matrix3x3 covariance) CylinderToGaussian(Vector3 direction, float startDistance, float endDistance, float radius, bool diagonalCovariance)
        {
            float distanceMean = (startDistance + endDistance) / 2;
            float radiusVariance = radius * radius / 4;
            float distanceVariance = (endDistance - startDistance) * (endDistance - startDistance) / 12;
            return LiftGaussian(direction, distanceMean, distanceVariance, radiusVariance, diagonalCovariance);
        }
        public static (Vector3 means, Matrix3x3 covariances)[] CastRay(float[] tVals, Vector3 origin, Vector3 direction, float radius, RayShape rayShape, bool diagonalCovariance = true)
        {
            int numSegments = tVals.Length - 1;
            (Vector3 mean, Matrix3x3 covariance)[] samples = new (Vector3, Matrix3x3)[numSegments - 1];
            for (int i = 0; i < numSegments - 1; i++)
            {
                float t0 = tVals[i];
                float t1 = tVals[i + 1];
                (Vector3 segmentMean, Matrix3x3 segmentCovariance) = rayShape switch
                {
                    RayShape.Conical => ConicalFrustumToGaussian(direction, t0, t1, radius, diagonalCovariance),
                    RayShape.Cylindrical => CylinderToGaussian(direction, t0, t1, radius, diagonalCovariance),
                    _ => throw new ArgumentException("Invalid Ray Shape")
                };
                samples[i].mean = segmentMean + origin;
                samples[i].covariance = segmentCovariance;
            }
            return samples;
        }
        public static Vector3[] IntegratedPositionalEncoding((Vector3 x, Matrix3x3 xCovariance) xCoordinate, int minDeg, int maxDeg, bool diag = true)
        {
            const int numDims = 3;
            Vector3 x = xCoordinate.x;
            int numFrequencies = maxDeg - minDeg;
            Vector3[] means = new Vector3[numFrequencies * 2];

            if (diag)
            {
                for (int i = 0; i < numFrequencies; i++)
                {
                    float scale = 1 << (i + minDeg);
                    Vector3 y = x * scale;
                    Vector3 yVar = xCoordinate.xCovariance.Diagonal * scale * scale;
                    for (int j = 0; j < numDims; j++)
                    {
                        means[2*i][j] = ExpectedSin(y[j], yVar[j]).mean;
                        means[2*i + 1][j] = ExpectedSin(y[j] + MathF.PI * 0.5f, yVar[j]).mean;
                    }
                }
            }
            else
            {
                for (int i = 0; i < numFrequencies; i++)
                {
                    float scale = (float)Math.Pow(2, i + minDeg);
                    Vector3 y = x * scale;
                    Matrix3x3 temp = xCoordinate.xCovariance * scale;
                    Matrix3x3 tempTranspose = temp.transpose;
                    Vector3 yVar = new(
                        Vector3.Dot(tempTranspose[0], temp[0]),
                        Vector3.Dot(tempTranspose[1], temp[1]),
                        Vector3.Dot(tempTranspose[2], temp[2]));
                    for (int j = 0; j < numDims; j++)
                    {
                        means[i][2 * j] = ExpectedSin(y[j], yVar[j]).mean;
                        means[i][2 * j + 1] = ExpectedSin(y[j] + MathF.PI * 0.5f, yVar[j]).mean;
                    }
                }
            }

            return means;
        }
        public static (Vector3 compositeRgb, float distance, float accumulation, float[] weights) VolumetricRendering((Vector3 color, float density)[] samples, float[] tVals, Vector3 direction, bool whiteBackground)
        {
            int numSamples = samples.Length;
            float[] alpha = new float[numSamples - 1];
            float[] transmittance = new float[numSamples];
            float[] weights = new float[numSamples - 1];
            Vector3 compRgb = Vector3.Zero;
            float acc = 0f;
            float weightedDistanceSum = 0f;
            for (int i = 0; i < numSamples - 1; i++)
            {
                alpha[i] = 1 - MathF.Exp(-samples[i].density * (tVals[i + 1] - tVals[i]) * direction.Length());
                transmittance[i] = i == 0 ? 1f : transmittance[i - 1] * (1 - alpha[i - 1]);
                weights[i] = alpha[i] * transmittance[i];
                compRgb += weights[i] * samples[i].color;
                acc += weights[i];
                weightedDistanceSum += weights[i] * (tVals[i] + tVals[i + 1]) / 2;
            }
            float distance = Math.Clamp(acc > 0 ? weightedDistanceSum / acc : float.PositiveInfinity, tVals[0], tVals[^1]);
            if (whiteBackground) compRgb += new Vector3(1f - acc);
            return (compRgb, distance, acc, weights);
        }
        public static (Vector3 compositeRgb, float distance, float accumulation, float[] alpha, float[] transmittance, float[] weights) CachedVolumetricRendering((Vector3 color, float density)[] samples, float[] tVals, Vector3 direction, bool whiteBackground)
        {
            int numSamples = samples.Length;
            float[] alpha = new float[numSamples - 1];
            float[] transmittance = new float[numSamples];
            float[] weights = new float[numSamples - 1];
            Vector3 compRgb = Vector3.Zero;
            float acc = 0f;
            float weightedDistanceSum = 0f;
            for (int i = 0; i < numSamples - 1; i++)
            {
                alpha[i] = 1 - MathF.Exp(-samples[i].density * (tVals[i + 1] - tVals[i]) * direction.Length());
                transmittance[i] = i == 0 ? 1f : transmittance[i - 1] * (1 - alpha[i - 1]);
                weights[i] = alpha[i] * transmittance[i];
                compRgb += weights[i] * samples[i].color;
                acc += weights[i];
                weightedDistanceSum += weights[i] * (tVals[i] + tVals[i + 1]) / 2;
            }
            float distance = Math.Clamp(acc > 0 ? weightedDistanceSum / acc : float.MaxValue, tVals[0], tVals[^1]);
            if (whiteBackground) compRgb += new Vector3(1f - acc);
            return (compRgb, distance, acc, alpha, transmittance, weights);
        }

        public static (Vector3 colorGradient, float densityGradient)[] VolumetricRenderingGradient(
            Vector3 compositeRgbGradient,
            float[] returnedAlpha,
            float[] returnedTransmittance,
            float[] returnedWeights,
            (Vector3 color, float density)[] samples,
            float[] tVals,
            Vector3 direction,
            bool whiteBackground)
        {
            int numSamples = samples.Length;
            Vector3[] colorGradients = new Vector3[numSamples];
            float[] densityGradients = new float[numSamples];
            float[] dLdWeights = new float[numSamples - 1];
            // Compute gradients with respect to weights and colors
            for (int i = 0; i < numSamples - 1; i++)
            {
                // dL/dWeights[i] = compositeRgbGradient ⋅ color[i]
                dLdWeights[i] = Vector3.Dot(compositeRgbGradient, samples[i].color);

                // dL/dColor[i] = compositeRgbGradient * weights[i]
                colorGradients[i] += compositeRgbGradient * returnedWeights[i];
            }

            // Handle white background case
            if (whiteBackground)
            {
                // dL/dAcc = - (dL/dCompRgb ⋅ 1)
                float dLdAcc = -(compositeRgbGradient.X + compositeRgbGradient.Y + compositeRgbGradient.Z);

                // dL/dWeights[i] += dL/dAcc (since acc = sum(weights))
                for (int i = 0; i < numSamples - 1; i++)
                {
                    dLdWeights[i] += dLdAcc;
                }
            }

            // Initialize gradients with respect to transmittance and alpha
            float[] dLdTransmittance = new float[numSamples];
            float[] dLdAlpha = new float[numSamples - 1];

            // Initialize dLdTransmittance to zero
            for (int i = 0; i < numSamples; i++)
            {
                dLdTransmittance[i] = 0f;
            }

            // Backpropagate through weights
            for (int i = 0; i < numSamples - 1; i++)
            {
                // weights[i] = alpha[i] * transmittance[i]
                dLdAlpha[i] += dLdWeights[i] * returnedTransmittance[i];
                dLdTransmittance[i] += dLdWeights[i] * returnedAlpha[i];
            }

            // Backpropagate through transmittance recursively
            for (int i = numSamples - 2; i >= 0; i--)
            {
                // transmittance[i + 1] = transmittance[i] * (1 - alpha[i])
                dLdTransmittance[i] += dLdTransmittance[i + 1] * (1 - returnedAlpha[i]);
                dLdAlpha[i] += -dLdTransmittance[i + 1] * returnedTransmittance[i];
            }

            // Compute gradient with respect to density
            float directionLength = direction.Length();

            for (int i = 0; i < numSamples - 1; i++)
            {
                float deltaT = tVals[i + 1] - tVals[i];
                float s = samples[i].density * deltaT * directionLength;

                // Compute exp(-s)
                float expNegS = 1 - returnedAlpha[i]; // Since returnedAlpha[i] = 1 - exp(-s)

                // Compute dAlpha/dDensity
                float dalphaDdensity = expNegS * deltaT * directionLength;

                // Compute dL/dDensity[i] = dL/dAlpha[i] * dAlpha/dDensity
                densityGradients[i] += dLdAlpha[i] * dalphaDdensity;
            }

            // The last sample does not contribute to the output gradients
            colorGradients[numSamples - 1] = Vector3.Zero;
            densityGradients[numSamples - 1] = 0f;

            // Combine gradients into output array
            (Vector3 colorGradient, float densityGradient)[] gradients = new (Vector3 colorGradient, float densityGradient)[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                gradients[i] = (colorGradients[i], densityGradients[i]);
            }

            return gradients;
        }
        public static (float[] tVals, (Vector3 mean, Matrix3x3 covariance)[] samples) SampleAlongRay(Random random, Vector3 origin, Vector3 direction, float radius, int numSamples, float near, float far, bool randomized,
            bool linearDisparity,
            RayShape rayShape)
        {
            float[] tVals = new float[numSamples + 1];
            for (int i = 0; i <= numSamples; i++) tVals[i] = (float)i / numSamples;

            if (linearDisparity)
                for (int i = 0; i <= numSamples; i++)
                    tVals[i] = 1f / (1f / near * (1f - tVals[i]) + 1f / far * tVals[i]);
            else
                for (int i = 0; i <= numSamples; i++) tVals[i] = near * (1f - tVals[i]) + far * tVals[i];

            if (!randomized) return (tVals, CastRay(tVals, origin, direction, radius, rayShape));
            float[] midpoints = new float[numSamples];
            for (int i = 0; i < numSamples; i++) midpoints[i] = 0.5f * (tVals[i] + tVals[i + 1]);
            for (int i = numSamples; i > 0; i--) tVals[i] = tVals[i - 1];
            for (int i = 1; i <= numSamples; i++) tVals[i] = midpoints[i - 1];
            for (int i = 0; i < numSamples; i++) tVals[i] += (tVals[i + 1] - tVals[i]) * random.NextSingle();
            return (tVals, CastRay(tVals, origin, direction, radius, rayShape));
        }

        // Resample Along Rays
        public static (float[] newTVals, (Vector3 mean, Matrix3x3 covariance)[] samples) ResampleAlongRay(
            Random random,
            Vector3 origin,
            Vector3 direction,
            float radius,
            float[] tVals,
            float[] weights,
            bool randomized,
            RayShape rayShape,
            float resamplePadding)
        {
            // Blurpool the weights
            float[] weightsPad = new float[weights.Length + 2];
            weightsPad[0] = weights[0];
            Array.Copy(weights, 0, weightsPad, 1, weights.Length);
            weightsPad[^1] = weights[^1];

            float[] weightsMax = new float[weightsPad.Length - 1];
            for (int i = 0; i < weightsMax.Length; i++)
            {
                weightsMax[i] = Math.Max(weightsPad[i], weightsPad[i + 1]);
            }

            float[] weightsBlur = new float[weightsMax.Length - 1];
            for (int i = 0; i < weightsBlur.Length; i++)
            {
                weightsBlur[i] = 0.5f * (weightsMax[i] + weightsMax[i + 1]) + resamplePadding;
            }
            // Resample
            float[] newTVals = MathHelpers.SortedPiecewiseConstantPDF(random, tVals, weightsBlur, tVals.Length, randomized);

            return (newTVals, CastRay(newTVals, origin, direction, radius, rayShape));
        }
    }

    public static class MathHelpers
    {
        const float Ln10 = 2.3025850929940456840179914546844f;
        public static float MseToPsnr(float mse) => -10.0f / Ln10 * MathF.Log(mse);
        public static float NextGaussian(this Random random) => MathF.Sqrt(-2.0f * MathF.Log(random.NextSingle())) * MathF.Sin(2.0f * MathF.PI * random.NextSingle());
        public static float PsnrToMse(float psnr) => MathF.Exp(-0.1f * Ln10 * psnr);
        public static float GlorotUniform(this Random rand, int inputDim, int outputDim) => MathF.Sqrt(6.0f / (inputDim + outputDim)) * (rand.NextSingle() * 2 - 1);

        public static float ComputeAvgError(float psnr, float ssim, float lpips)
        {
            float mse = PsnrToMse(psnr);
            float dssim = MathF.Sqrt(1 - ssim);
            return MathF.Exp(
                (MathF.Log(mse) + MathF.Log(dssim) + MathF.Log(lpips)) / 3.0f
            );
        }
        public static Vector3[,] ComputeSsim(VectorImage img0, VectorImage img1, float maxVal,
                                    int filterSize = 11, float filterSigma = 1.5f,
                                    float k1 = 0.01f, float k2 = 0.03f)
        {

            // Construct a 1D Gaussian blur filter
            VectorImage filt = CreateGaussianFilter(filterSize, filterSigma);

            // Apply filter to both images
            VectorImage mu0 = VectorImage.Convolve(img0, filt);
            VectorImage mu1 = VectorImage.Convolve(img1, filt);

            VectorImage mu00 = mu0 * mu0;
            VectorImage mu11 = mu1 * mu1;
            VectorImage mu01 = mu0 * mu1;
            VectorImage sigma00 = VectorImage.Convolve(img0 * img0, filt) - mu00;
            VectorImage sigma11 = VectorImage.Convolve(img1 * img1, filt) - mu11;
            VectorImage sigma01 = VectorImage.Convolve(img0 * img1, filt) - mu01;

            // Clip the variances and covariances to valid values
            sigma00 = VectorImage.ApplyOperation(sigma00, (vec3) => new (MathF.Max(vec3.X, 0), MathF.Max(vec3.Y, 0), MathF.Max(vec3.Z, 0)));
            sigma11 = VectorImage.ApplyOperation(sigma11, (vec3) => new (MathF.Max(vec3.X, 0), MathF.Max(vec3.Y, 0), MathF.Max(vec3.Z, 0)));
            sigma01 = VectorImage.ApplyOperation(sigma01, (vec3) => new (MathF.Max(vec3.X, 0), MathF.Max(vec3.Y, 0), MathF.Max(vec3.Z, 0)));

            float c1 = MathF.Pow(k1 * maxVal, 2);
            float c2 = MathF.Pow(k2 * maxVal, 2);

            Vector3[,] ssimMap = new Vector3[img0.Width, img0.Height];

            for (int i = 0; i < img0.Width; i++)
            {
                for (int j = 0; j < img0.Height; j++)
                {
                    Vector3 numerator = (mu01[i, j] * 2 + Vector3.One * c1) * (sigma01[i,j] * 2 + Vector3.One * c2);
                    Vector3 denominator = (mu00[i, j] + mu11[i, j] + new Vector3(c1)) * (sigma00[i, j] + sigma11[i, j] + Vector3.One * c2);
                    ssimMap[i, j] = numerator / denominator;
                }
            }
            return ssimMap;
        }

        public static float ComputeSsimAverage(VectorImage img0, VectorImage img1, float maxVal,
            int filterSize = 11, float filterSigma = 1.5f,
            float k1 = 0.01f, float k2 = 0.03f)
        {
            Vector3[,] ssimMap = ComputeSsim(img0, img1, maxVal, filterSize, filterSigma, k1, k2);
            float ssimSum = 0;
            for (int i = 0; i < img0.Width; i++)
            for (int j = 0; j < img0.Height; j++)
                ssimSum += (ssimMap[i, j].X + ssimMap[i, j].Y + ssimMap[i, j].Z) / 3;
            return ssimSum / (img0.Width * img0.Height);
        }
        private static VectorImage CreateGaussianFilter(int size, float sigma)
        {
            VectorImage filter = new(size, size);
            Vector3 sum = Vector3.Zero;
            int halfSize = size / 2;
            for (int i = 0; i < size; i++)
            for(int j = 0; j < size; j++)
            {
                float x = i - halfSize;
                float y = j - halfSize;
                filter[i, j] = Vector3.One * MathF.Exp(-(x * x + y * y) / (2 * sigma * sigma));
                sum += filter[i, j];
            }
            for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                filter[i, j] /= sum;
            return filter;
        }
        public static float LinearToSrgb(float linear) => linear <= 0.0031308f ? 12.92f * linear : 1.055f * MathF.Pow(linear, 1f / 2.4f) - 0.055f;

        public static float SrgbToLinear(float srgb) => srgb <= 0.04045f ? srgb / 12.92f : MathF.Pow((srgb + 0.055f) / 1.055f, 2.4f);
        public static float LearningRateDecay(int step, float learningRateInit, float learningRateFinal, int maxSteps, int learningRateDelaySteps = 0, float learningRateDelayMult = 1f)
        {
            float delayRate = 1f;
            if (learningRateDelaySteps > 0)
            {
                float delayProgress = Math.Clamp((float)step / learningRateDelaySteps, 0f, 1f);
                delayRate = learningRateDelayMult + (1f - learningRateDelayMult) *
                    MathF.Sin(0.5f * MathF.PI * delayProgress);
            }
            float t = Math.Clamp((float)step / maxSteps, 0f, 1f);
            float logLerp = MathF.Exp(
                MathF.Log(learningRateInit) * (1 - t) + MathF.Log(learningRateFinal) * t
            );

            return delayRate * logLerp;
        }
        public static float[] SortedPiecewiseConstantPDF(
        Random rand,
        float[] tVals,
        float[] weights,
        int numSamples,
        bool randomized)
        {
            int numBins = weights.Length;

            // Step 1: Adjust weights to avoid NaNs
            float eps = 1e-5f;
            float weightSum = weights.Sum();

            float padding = MathF.Max(0f, eps - weightSum);
            if (padding > 0f)
            {
                float paddingPerBin = padding / numBins;
                weights = weights.Select(w => w + paddingPerBin).ToArray();
                weightSum += padding;
            }

            // Step 2: Compute PDF
            float[] pdf = weights.Select(w => w / weightSum).ToArray();

            // Step 3: Compute CDF
            float[] cdfPartialSums = pdf.Take(numBins - 1).ToArray(); // pdf[..., :-1]
            float[] cdfCumulative = new float[numBins - 1];
            float cumulativeSum = 0f;
            for (int i = 0; i < cdfPartialSums.Length; i++)
            {
                cumulativeSum += cdfPartialSums[i];
                cdfCumulative[i] = MathF.Min(1f, cumulativeSum);
            }

            // Concatenate zeros at the start and ones at the end
            float[] cdf = new float[numBins + 1];
            cdf[0] = 0f;
            Array.Copy(cdfCumulative, 0, cdf, 1, cdfCumulative.Length);
            cdf[numBins] = 1f;

            // Step 4: Draw uniform samples
            float[] u = new float[numSamples];
            float s1 = 1f / numSamples;
            for (int i = 0; i < numSamples; i++)
            {
                u[i] = MathF.Min(i * s1 + (rand.NextSingle() * (s1 - 1e-7f)), 1f - 1e-7f);
            }

            // Step 5: Identify intervals and compute samples
            float[] samples = new float[numSamples];
            for (int s = 0; s < numSamples; s++)
            {
                // Find the interval index where u[s] falls into
                int idx = Array.BinarySearch(cdf, u[s]);
                if (idx < 0)
                {
                    idx = ~idx - 1; // Get the index of the lower bound
                }
                idx = Math.Clamp(idx, 0, numBins - 1);

                // Get corresponding tVals and cdf values
                float binsG0 = tVals[idx];
                float binsG1 = tVals[idx + 1];
                float cdfG0 = cdf[idx];
                float cdfG1 = cdf[idx + 1];

                // Compute the sample using linear interpolation
                float denom = cdfG1 - cdfG0;

                // Handle potential division by zero (equivalent to jnp.nan_to_num)
                float t = denom > 0f ? (u[s] - cdfG0) / denom : 0f;

                t = Math.Clamp(t, 0f, 1f);
                samples[s] = binsG0 + t * (binsG1 - binsG0);
            }

            return samples;
        }
    }

    public struct VectorImage(int width, int height)
    {
        public int Width = width;
        public int Height = height;
        private Vector3[,] _pixels = new Vector3[width, height];
        public Vector3 this[int x, int y]
        {
            get => _pixels[x, y];
            set => _pixels[x, y] = value;
        }
        public static VectorImage operator +(VectorImage a, VectorImage b)
        {
            if (a.Width != b.Width || a.Height != b.Height)
                throw new ArgumentException("Images must have the same dimensions for addition.");
            VectorImage result = new(a.Width, a.Height);
            for (int x = 0; x < a.Width; x++)
            for (int y = 0; y < a.Height; y++)
                result[x, y] = a[x, y] + b[x, y];
            return result;
        }
        public static VectorImage operator -(VectorImage a, VectorImage b)
        {
            if (a.Width != b.Width || a.Height != b.Height)
                throw new ArgumentException("Images must have the same dimensions for subtraction.");
            VectorImage result = new(a.Width, a.Height);
            for (int x = 0; x < a.Width; x++)
            for (int y = 0; y < a.Height; y++)
                result[x, y] = a[x, y] - b[x, y];
            return result;
        }
        public static VectorImage operator *(VectorImage a, float b)
        {
            VectorImage result = new(a.Width, a.Height);
            for (int x = 0; x < a.Width; x++)
            for (int y = 0; y < a.Height; y++)
                result[x, y] = a[x, y] * b;
            return result;
        }
        public static VectorImage operator *(VectorImage a, VectorImage b)
        {
            if (a.Width != b.Width || a.Height != b.Height)
                throw new ArgumentException("Images must have the same dimensions for multiplication.");
            VectorImage result = new(a.Width, a.Height);
            for (int x = 0; x < a.Width; x++)
            for (int y = 0; y < a.Height; y++)
                result[x, y] = a[x, y] * b[x, y];
            return result;
        }

        public static VectorImage Convolve(VectorImage image, VectorImage kernel)
        {
            int kernelWidth = kernel.Width;
            int kernelHeight = kernel.Height;
            int padX = kernelWidth / 2;
            int padY = kernelHeight / 2;
            VectorImage paddedImage = new(image.Width + 2 * padX, image.Height + 2 * padY);
            for (int x = 0; x < image.Width; x++)
            for (int y = 0; y < image.Height; y++) paddedImage[x + padX, y + padY] = image[x, y];
            VectorImage result = new(image.Width, image.Height);
            for (int x = 0; x < image.Width; x++)
            for (int y = 0; y < image.Height; y++)
            {
                Vector3 sum = Vector3.Zero;
                for (int kx = 0; kx < kernelWidth; kx++)
                for (int ky = 0; ky < kernelHeight; ky++)
                {
                    int imageX = x + kx;
                    int imageY = y + ky;
                    sum += paddedImage[imageX, imageY] * kernel[kx, ky];
                }
                result[x, y] = sum;
            }
            return result;
        }
        public static VectorImage operator /(VectorImage image1, VectorImage image2) 
        {
            if (image1.Width != image2.Width || image1.Height != image2.Height)
                throw new ArgumentException("Images must have the same dimensions for division.");
            VectorImage result = new(image1.Width, image1.Height);
            for (int x = 0; x < image1.Width; x++)
            for (int y = 0; y < image1.Height; y++)
                result[x, y] = image1[x, y] / image2[x, y];
            return result;
        }
        public static VectorImage ApplyOperation(VectorImage image, Func<Vector3, Vector3> operation)
        {
            VectorImage result = new(image.Width, image.Height);
            for (int x = 0; x < image.Width; x++)
            for (int y = 0; y < image.Height; y++)
                result[x, y] = operation(image[x, y]);
            return result;
        }
    }

}
