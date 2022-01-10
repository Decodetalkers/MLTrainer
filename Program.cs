// See https://aka.ms/new-console-template for more info
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
//using System;
//using System.Linq;
//using System.Collections.Generic;
namespace Gisture
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext(); // MLContext?
            var data = context.Data.LoadFromTextFile<GistureData>("./csvs/persion8.csv", hasHeader: false, separatorChar: ',');//IdataView?

            var middle_data = context.Data
                .CreateEnumerable<GistureData>(data, reuseRowObject: false);
            var final_data = Convert_to_DataPoints(middle_data);
            var data2 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion1.csv", hasHeader: false, separatorChar: ',');//IdataView?

            var middle_data2 = context.Data
                .CreateEnumerable<GistureData>(data2, reuseRowObject: false);
            var final_data2 = Convert_to_DataPoints(middle_data2);
            
			var options = new LinearSvmTrainer.Options
            {
                LabelColumnName = "Label",
                BatchSize = 10,
                PerformProjection = true,
                NumberOfIterations = 10,
            };
            var pipeline = context.Transforms.Concatenate("Features", new[] { "ThumbIndex", "IndexAngle", "MiddleAngle", "RingAngle", "PinkyAngle", "ThumbIndex", "IndexMiddle" })
                .Append(context.BinaryClassification.Trainers.LinearSvm(options));
            var model = pipeline.Fit(final_data);
            var transform_predicts = model.Transform(final_data2);
            var predicts = context.Data
                .CreateEnumerable<Prediction>(transform_predicts,
                reuseRowObject: false).ToList();

            foreach (var item in predicts)
            {
                Console.WriteLine($"{item.Label},{item.PredictedLabel}");
            }
			var metrics = context.BinaryClassification
				.EvaluateNonCalibrated(transform_predicts);
			PrintMetrics(metrics);

        }
        public class Prediction
        {
            public bool Label { get; set; }
            public bool PredictedLabel { get; set; }
        }
        private static IDataView Convert_to_DataPoints(IEnumerable<GistureData> input)
        {
            IEnumerable<DataPoint> Middle(IEnumerable<GistureData> input)
            {
                foreach (var item in input)
                {
                    yield return new DataPoint
                    {
                        Label = item.Kind == 1,
                        ThumbAngle = item.ThumbAngle,
                        IndexAngle = item.IndexAngle,
                        MiddleAngle = item.MiddleAngle,
                        RingAngle = item.RingAngle,
                        PinkyAngle = item.PinkyAngle,
                        ThumbIndex = item.ThumbIndex,
                        IndexMiddle = item.IndexMiddle,
                    };
                }
            };
			var context = new MLContext();
            return context.Data.LoadFromEnumerable(Middle(input));
        }
		private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}
