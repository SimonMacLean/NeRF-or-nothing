using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace ScratchNerf
{
    public enum Split
    {
        Train,
        Test
    }
    public static class DatasetFactory
    {
        public static Dataset CreateDataset(Split split, string dataDir)
        {
            return Config.DatasetLoader switch
            {
                DatasetType.Multicam => new Multicam(split, dataDir),
                DatasetType.Blender => new Blender(split, dataDir),
                DatasetType.LLFF => new LLFF(split, dataDir),
                _ => throw new ArgumentOutOfRangeException()
            };
        }
    }
    public abstract class Dataset
    {
        protected float near = Config.Near;
        protected float far = Config.Far;
        protected Split split;
        protected string dataDir;
        public int numExamples;
        protected float focalLength;
        protected int w;
        protected int h;
        protected Ray[,,] rays;
        protected Dictionary<string, object> metadata;
        protected Bitmap[] images;
        protected Vector3[,,] pixels;
        private Random rand = new();
        protected (Matrix3x3 rotationMatrix, Vector3 translationVector)[] camToWorlds;
        protected (Ray[] rays, Vector3[] pixels) currentBatch;
        public static Vector3[,] AsFloats(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;
            Vector3[,] result = new Vector3[height, width];

            BitmapData bitmapData = image.LockBits(new Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* scan0 = (byte*)bitmapData.Scan0.ToPointer();
                    for (int y = 0; y < height; y++)
                    {
                        byte* row = scan0 + (y * bitmapData.Stride);
                        for (int x = 0; x < width; x++) 
                            result[y, x] = new Vector3(row[x * 3], row[x * 3 + 1], row[x * 3 + 2]) / 255f;
                    }
                }
            }
            finally
            {
                image.UnlockBits(bitmapData);
            }

            return result;
        }
        protected Dataset(Split split, string dataDir)
        {
            this.split = split;
            this.dataDir = dataDir;
            switch (split)
            {
                case Split.Train:
                    TrainInit();
                    break;
                case Split.Test:
                    TestInit();
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        protected void TrainInit()
        {
            LoadRenderings();
            GenerateRays();
            pixels = new Vector3[images.Length, h, w];
            for (int i = 0; i < images.Length; i++)
            {
                Vector3[,] imagePixels = AsFloats(images[i]);
                for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    pixels[i, y, x] = imagePixels[y, x];
            }
            currentBatch = NextTrain();
        }
        protected void TestInit()
        {
            throw new NotImplementedException();
        }
        protected virtual void GenerateRays()
        {
            // Generate camera directions
            Vector3[,] cameraDirs = new Vector3[h, w];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float dirX = (x - w * 0.5f + 0.5f) / focalLength;
                    float dirY = -(y - h * 0.5f + 0.5f) / focalLength;
                    cameraDirs[y, x] = new Vector3(dirX, dirY, -1);
                }
            }

            int numImages = camToWorlds.Length;
            Ray[,,] rays = new Ray[numImages, h, w];

            for (int imageIndex = 0; imageIndex < numImages; imageIndex++)
            {
                Matrix3x3 rotation = camToWorlds[imageIndex].rotationMatrix;
                Vector3 translation = camToWorlds[imageIndex].translationVector;

                // Calculate directions
                Vector3[,] directions = new Vector3[h, w];
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        directions[y, x] = rotation * cameraDirs[y, x];
                    }
                }

                // Calculate radii
                float[,] radii = new float[h, w];
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int nextX = x < w - 1 ? x + 1 : x;
                        Vector3 diff = directions[y, x] - directions[y, nextX];
                        radii[y, x] = diff.Length() * 2 / MathF.Sqrt(12);
                    }
                }

                // Generate rays
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        Vector3 direction = directions[y, x];
                        rays[imageIndex, y, x] = new Ray
                        {
                            Direction = direction,
                            Origin = translation,
                            ViewDir = Vector3.Normalize(direction),
                            LossMult = 1,
                            Near = near,
                            Far = far,
                            Radius = radii[y, x]
                        };
                    }
                }
            }

            this.rays = rays;
        }
        public (Ray[] rays, Vector3[] pixels) Next()
        {
            (Ray[] rays, Vector3[] pixels) x = currentBatch;
            currentBatch = NextTrain();
            return x;
        }

        public (Ray[] rays, Vector3[] pixels) Peek() => currentBatch;
        public abstract void LoadRenderings();

        private (Ray[] rays, Vector3[] pixels) NextTrain()
        {
            (int, int, int)[] rayIndices = new (int, int, int)[Config.BatchSize];
            (int, int, int)[] possibleIndices = new (int, int, int)[rays.Length];
            for (int i = 0; i < Config.BatchSize; i++)
            {
                (int, int, int) indexToCheck = (rand.Next(rays.Length), rand.Next(h), rand.Next(w));
                int flattenedIndex = indexToCheck.Item1 * h * w + indexToCheck.Item2 * w + indexToCheck.Item3;
                rayIndices[i] = possibleIndices[flattenedIndex] == (0,0,0) ? indexToCheck : (possibleIndices[flattenedIndex].Item1 - 1, possibleIndices[flattenedIndex].Item2, possibleIndices[flattenedIndex].Item3);
                (int, int, int) unflattenedI = (flattenedIndex / (h * w), (flattenedIndex % (h * w)) / w, flattenedIndex % w);
                possibleIndices[flattenedIndex] = (unflattenedI.Item1 + 1, unflattenedI.Item2, unflattenedI.Item3);
            }
            return (rayIndices.Select((i) => rays[i.Item1, i.Item2, i.Item3]).ToArray(), rayIndices.Select((i) => pixels[i.Item1, i.Item2, i.Item3]).ToArray());
        }
    }
    class Multicam(Split split, string dataDir) : Dataset(split, dataDir)
    {
        Dictionary<string, object> metadata;
        public override void LoadRenderings() => throw new NotImplementedException();
    }
    class Blender(Split split, string dataDir) : Dataset(split, dataDir)
    {
        public override void LoadRenderings() => throw new NotImplementedException();
    }
    class LLFF(Split split, string dataDir) : Dataset(split, dataDir)
    {
        private int resolution;
        public override void LoadRenderings()
        {
            string imgDirSuffix = "";
            int factor = Math.Max(1, Config.Factor);
            if (Config.Factor > 0) imgDirSuffix = $"_{Config.Factor}";
            string imgDir = Path.Join(dataDir, $"images{imgDirSuffix}");
            string[] imgFiles = Directory.GetFiles(imgDir).Where((fileName) =>
                fileName.ToLower().EndsWith(".jpg") || fileName.ToLower().EndsWith(".png")).ToArray();
            Bitmap[] images = imgFiles.Select((fileName) => new Bitmap(fileName)).ToArray();
            string poseFile = Path.Join(dataDir, "poses_bounds.csv");
            float[][] posesArr = File.ReadAllLines(poseFile).Select((line) =>
                line.Split(',').Select((s) => float.Parse(s)).ToArray()).ToArray();
            (Matrix3x3 rotation, Vector3 translation)[] poses = new (Matrix3x3, Vector3)[posesArr.Length];
            for (int i = 0; i < posesArr.Length; i++)
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                    poses[i].rotation[j, k] = posesArr[i][4 * j + k];
                poses[i].translation[j] = posesArr[i][4 * j + 3];
            }

            float[,] bounds = new float[posesArr.Length, 2];
            for (int i = 0; i < posesArr.Length; i++)
            for (int j= 0; j < 2; j++)
                bounds[i, j] = posesArr[i][12 + j];
            if (posesArr.Length != images.Length)
                throw new Exception("Number of images and poses do not match.");
            float scale = 1 / (bounds.Cast<float>().Min() + 0.75f);
            for (int i = 0; i < posesArr.Length; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    (poses[i].rotation[j, 0], poses[i].rotation[j, 1]) = (poses[i].rotation[j, 1], -poses[i].rotation[j, 0]);
                    bounds[i,j] *= scale;
                }
                poses[i].translation *= scale;
            }
            RecenterPoses(poses);
            if (Config.Spherify)
                GenerateSphericalPoses(poses, bounds);
            else if (split == Split.Test)
                GenerateSpiralPoses(poses, bounds);
            camToWorlds = poses;
            this.images = images;
            w = images[0].Width;
            h = images[0].Height;
            resolution = w * h;
            numExamples = images.Length;
            focalLength = w * factor;
        }

        protected override void GenerateRays()
        {
            base.GenerateRays();
            if (Config.Spherify) return;
            for (int i = 0; i < rays.GetLength(0); i++)
            for (int j = 0; j < rays.GetLength(1); j++)
            for (int k = 0; k < rays.GetLength(2); k++)
                (rays[i, j, k].Origin, rays[i, j, k].Direction) = ConvertToNdc(rays[i, j, k].Origin, rays[i, j, k].Direction, focalLength, w, h);
            float[,,] distances = new float[rays.GetLength(0), rays.GetLength(1), rays.GetLength(2)];

            for (int b = 0; b < rays.GetLength(0); b++)
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float dx = x < w - 1
                    ? Vector3.Distance(rays[b, y, x].Origin, rays[b, y, x + 1].Origin)
                    : Vector3.Distance(rays[b, y, x - 1].Origin, rays[b, y, x].Origin);
                float dy = y < h - 1
                    ? Vector3.Distance(rays[b, y, x].Origin, rays[b, y + 1, x].Origin)
                    : Vector3.Distance(rays[b, y - 1, x].Origin, rays[b, y, x].Origin);
                distances[b, y, x] = MathF.Sqrt(dx * dx + dy * dy);
                rays[b, y, x].Radius = distances[b, y, x] / MathF.Sqrt(12);
                rays[b, y, x].Near = near;
                rays[b, y, x].Far = far;
            }
        }

        public (Vector3 origin, Vector3 direction) ConvertToNdc(Vector3 origin, Vector3 direction, float focal, float w, float h, float near = 1f)
        {
            float t = -(near + origin.Z) / direction.Z;
            origin += t * direction;
            float o0 = -(2 * focal / w) * (origin.X / origin.Z);
            float o1 = -(2 * focal / h) * (origin.Y / origin.Z);
            float o2 = 1 + 2 * near / origin.Z;
            float d0 = -(2 * focal / w) * (direction.X / direction.Z - origin.X / origin.Z);
            float d1 = -(2 * focal / h) * (direction.Y / direction.Z - origin.Y / origin.Z);
            float d2 = -2 * near / origin.Z;
            Vector3 newOrigin = new(o0, o1, o2);
            Vector3 newDirection = new(d0, d1, d2);
            return (newOrigin, newDirection);
        }
        private void RecenterPoses((Matrix3x3 rotation, Vector3 translation)[] poses)
        {
            (Matrix3x3 rotation, Vector3 translation) average = (poses.Select((p) => p.rotation).Aggregate((a, b) => a + b) / poses.Length,
                poses.Select((p) => p.translation).Aggregate((a, b) => a + b) / poses.Length);
            (Matrix3x3 rotation, Vector3 translation) averageInverse = (average.rotation.transpose, -average.rotation.transpose * average.translation);
            for (int i = 0; i < poses.GetLength(0); i++)
            {
                poses[i].translation -= poses[i].rotation * averageInverse.translation;
                poses[i].rotation *= averageInverse.rotation;
            }
        }
        private void GenerateSphericalPoses((Matrix3x3 rotation, Vector3 translation)[] poses, float[,] bounds) => throw new NotImplementedException();
        private void GenerateSpiralPoses((Matrix3x3 rotation, Vector3 translation)[] poses, float[,] bounds) => throw new NotImplementedException();
    }
}
