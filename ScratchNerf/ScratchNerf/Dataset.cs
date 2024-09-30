using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using 

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
        protected float focalLength;
        protected int w;
        protected int h;
        protected Ray[] rays;
        protected Dictionary<string, object> metadata;
        protected Bitmap[] images;
        protected Vector3[] pixels;
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
            pixels = images.SelectMany((bmp) => AsFloats(bmp).Cast<Vector3>().ToArray()).ToArray();
            currentBatch = NextTrain();
        }
        protected void TestInit()
        {
            throw new NotImplementedException();
        }
        protected void GenerateRays()
        {
            Vector3[,] cameraDirs = new Vector3[h, w];
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    float x = (j - w * 0.5f + 0.5f) / focalLength;
                    float y = -(i - h * 0.5f + 0.5f) / focalLength;
                    cameraDirs[i, j] = new Vector3(x, y, -1);
                }
            }
            int numImages = camToWorlds.Length;
            Ray[] rays = new Ray[h * w * numImages];
            for (int iImage = 0; iImage < numImages; iImage++)
            {
                Matrix3x3 rotation = camToWorlds[iImage].rotationMatrix;
                Vector3 translation = camToWorlds[iImage].translationVector;
                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        int index = iImage * h * w + i * w + j;
                        int indexToCalc = j < w - 1 ? j : j - 1;
                        Vector3 diff = rotation * cameraDirs[i, indexToCalc] - rotation * cameraDirs[i, indexToCalc + 1];
                        float radius = diff.Length() / MathF.Sqrt(3);

                        rays[index] = new Ray()
                        {
                            Direction = rotation * cameraDirs[i, j],
                            Origin = translation,
                            ViewDir = Vector3.Normalize(rotation * cameraDirs[i, j]),
                            LossMult = 1,
                            Near = near,
                            Far = far,
                            Radius = radius
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
            int[] rayIndices = new int[Config.BatchSize];
            int[] possibleIndices = new int[rays.Length];
            for (int i = 0; i < Config.BatchSize; i++)
            {
                int indexToCheck = rand.Next(rays.Length - i);
                rayIndices[i] = possibleIndices[indexToCheck] == 0 ? indexToCheck : possibleIndices[indexToCheck] - 1;
                possibleIndices[indexToCheck] = i + 1;
            }
            return (rayIndices.Select((i) => rays[i]).ToArray(), rayIndices.Select((i) => pixels[i]).ToArray());
        }
    }
    class Multicam(Split split, string dataDir) : Dataset(split, dataDir)
    {
        Dictionary<string, object> metadata;
        public override void LoadRenderings()
        {
        }
    }
    class Blender(Split split, string dataDir) : Dataset(split, dataDir)
    {
        public override void LoadRenderings()
        {
            throw new NotImplementedException();
        }
    }
    class LLFF(Split split, string dataDir) : Dataset(split, dataDir)
    {
        public override void LoadRenderings()
        {
            string imgDirSuffix = "";
            int factor = Math.Max(1, Config.Factor);
            if (Config.Factor > 0) imgDirSuffix = $"_{Config.Factor}";
            string imgDir = Path.Join(dataDir, $"images{imgDirSuffix}");
            string[] imgFiles = Directory.GetFiles(imgDir).Where((fileName) =>
                fileName.ToLower().EndsWith(".jpg") || fileName.ToLower().EndsWith(".png")).ToArray();
            Bitmap[] images = imgFiles.Select((fileName) => new Bitmap(fileName)).ToArray();
            string poseFile = Path.Join(dataDir, "poses_bounds.npy");

        }
    }
}
