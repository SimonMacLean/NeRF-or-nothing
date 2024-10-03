using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace ScratchNerf
{
    public class BinDataset(string file)
    {
        const int BatchSize = 1024;
        private (Ray[] rays, Vector3[] pixels)? _currentBatch;
        private readonly Random _rng = new();
        private readonly int _numSamples = (int)new FileInfo(file).Length / 64;

        public (Ray[] rays, Vector3[] pixels) Peek()
        {
            return _currentBatch ??= LoadBatch();
        }
        public (Ray[] rays, Vector3[] pixels) Next()
        {
            _currentBatch = LoadBatch();
            return _currentBatch.Value;
        }

        private (Ray[] rays, Vector3[] pixels) LoadBatch()
        {
            Ray[] rays = new Ray[BatchSize];
            Vector3[] pixels = new Vector3[BatchSize];
            using (FileStream fs = new(file, FileMode.Open, FileAccess.Read))
                for (int i = 0; i < BatchSize; i++)
                {
                    int currentSample = _rng.Next(_numSamples);
                    long startPosition = currentSample * 64;
                    byte[] data = new byte[64];
                    fs.Seek(startPosition, SeekOrigin.Begin);
                    if(fs.Read(data, 0, 64) != 64)
                        throw new Exception("Failed to read data from file");
                    float[] floatData = new float[16];
                    Buffer.BlockCopy(data, 0, floatData, 0, 64);
                    rays[i] = new Ray(new Vector3(floatData[0], floatData[1], floatData[2]),
                        new Vector3(floatData[3], floatData[4], floatData[5]),
                        new Vector3(floatData[6], floatData[7], floatData[8]), floatData[9], floatData[10],
                        floatData[11], floatData[12]);
                    pixels[i] = new Vector3(floatData[13], floatData[14], floatData[15]);
                }
            return (rays, pixels);
        }
    }
}
