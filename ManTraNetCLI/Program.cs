using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        string modelPath = args.Length > 0 ? args[0] : Path.Combine("pretrained_weights", "ManTraNet_Ptrain4.onnx");
        string csvPath = args.Length > 1 ? args[1] : Path.Combine("data", "samplePairs.csv");
        using var runner = new ManTraNetRunner(modelPath);
        float forgedSum = 0f;
        float origSum = 0f;
        int count = 0;
        foreach (var line in File.ReadLines(csvPath))
        {
            var parts = line.Split(',');
            if (parts.Length != 2) continue;
            string forged = Path.Combine("data", parts[0]);
            string orig = Path.Combine("data", parts[1]);
            forgedSum += runner.Predict(forged);
            origSum += runner.Predict(orig);
            count++;
            if (count >= 3) break;
        }
        Console.WriteLine($"Mean forged score: {forgedSum / count:F6}");
        Console.WriteLine($"Mean original score: {origSum / count:F6}");
    }
}
