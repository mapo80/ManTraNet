using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Advanced;
using System;
using System.IO;
using System.Linq;

public class ManTraNetRunner : IDisposable
{
    private readonly InferenceSession _session;
    public ManTraNetRunner(string modelPath)
    {
        _session = new InferenceSession(modelPath);
    }

    public float Predict(string imagePath)
    {
        using Image<Rgb24> img = Image.Load<Rgb24>(imagePath);
        img.Mutate(x => x.Resize(256, 256));
        float[] data = new float[256 * 256 * 3];
        int idx = 0;
        for (int y = 0; y < 256; y++)
        {
            Span<Rgb24> row = img.DangerousGetPixelRowMemory(y).Span;
            for (int x = 0; x < 256; x++)
            {
                data[idx++] = row[x].R / 255f * 2f - 1f;
                data[idx++] = row[x].G / 255f * 2f - 1f;
                data[idx++] = row[x].B / 255f * 2f - 1f;
            }
        }
        var tensor = new DenseTensor<float>(data, new[] {1, 256, 256, 3});
        var inputs = new List<NamedOnnxValue>{ NamedOnnxValue.CreateFromTensor("img_in", tensor) };
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();
        return output.Average();
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
