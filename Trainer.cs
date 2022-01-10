using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Gisture
{
    using Model = TransformerChain<BinaryPredictionTransformer<LinearBinaryModelParameters>>;
    class Trainers
    {
        public MLContext context;
        public Model[] models;
        public Trainers()
        {
            context = new MLContext();
            var data1 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion1.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var data2 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion2.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var data3 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion3.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var data4 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion4.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var data5 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion5.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var data6 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion6.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var data7 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion7.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var data8 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion8.csv", hasHeader: false, separatorChar: ',');//IdataView?

            var middle_data1 = context.Data
                .CreateEnumerable<GistureData>(data1, reuseRowObject: false);
            var middle_data2 = context.Data
                .CreateEnumerable<GistureData>(data2, reuseRowObject: false);
            var middle_data3 = context.Data
                .CreateEnumerable<GistureData>(data3, reuseRowObject: false);
            var middle_data4 = context.Data
                .CreateEnumerable<GistureData>(data4, reuseRowObject: false);
            var middle_data5 = context.Data
                .CreateEnumerable<GistureData>(data5, reuseRowObject: false);
            var middle_data6 = context.Data
                .CreateEnumerable<GistureData>(data6, reuseRowObject: false);
            var middle_data7 = context.Data
                .CreateEnumerable<GistureData>(data7, reuseRowObject: false);
            var middle_data8 = context.Data
                .CreateEnumerable<GistureData>(data8, reuseRowObject: false);
            var final_data0 = middle_data1
                .Union(middle_data2)
                .Union(middle_data3)
                .Union(middle_data3)
                .Union(middle_data4)
                .Union(middle_data5)
                .Union(middle_data6)
                .Union(middle_data7)
                .Union(middle_data8);

            var final_data1 = Convert_to_DataPoints(input: final_data0, kind: 1);
            var final_data2 = Convert_to_DataPoints(input: final_data0, kind: 2);
            var final_data3 = Convert_to_DataPoints(input: final_data0, kind: 3);
            var final_data4 = Convert_to_DataPoints(input: final_data0, kind: 4);
            var final_data5 = Convert_to_DataPoints(input: final_data0, kind: 5);
            var final_data6 = Convert_to_DataPoints(input: final_data0, kind: 6);
            var final_data7 = Convert_to_DataPoints(input: final_data0, kind: 7);
            var final_data8 = Convert_to_DataPoints(input: final_data0, kind: 8);

            var options = new LinearSvmTrainer.Options
            {
                LabelColumnName = "Label",
                BatchSize = 10,
                PerformProjection = true,
                NumberOfIterations = 10,
            };
            var pipeline = context.Transforms.Concatenate("Features", new[] { "ThumbIndex", "IndexAngle", "MiddleAngle", "RingAngle", "PinkyAngle", "ThumbIndex", "IndexMiddle" })
                .Append(context.BinaryClassification.Trainers.LinearSvm(options));
            var model1 = pipeline.Fit(final_data1);
            var model2 = pipeline.Fit(final_data2);
            var model3 = pipeline.Fit(final_data3);
            var model4 = pipeline.Fit(final_data4);
            var model5 = pipeline.Fit(final_data5);
            var model6 = pipeline.Fit(final_data6);
            var model7 = pipeline.Fit(final_data7);
            var model8 = pipeline.Fit(final_data8);
            models = new Model[] { model1, model2, model3, model4, model5, model6, model7, model8 };

        }
        public int Predict(DataPoint input)
        {
            IEnumerable<DataPoint> IterCreator(DataPoint input)
            {
                yield return input;
            }
            for (int i = 0; i < 8; ++i)
            {
                var data = context.Data.LoadFromEnumerable(IterCreator(input));
                var predicts = models[i].Transform(data);
                var predict = context.Data
                    .CreateEnumerable<Prediction>(predicts,
                            reuseRowObject: false).ToList()[0];
                if (predict.PredictedLabel)
                {
                    return i + 1;
                }
            }
            return 0;
        }
        public IDataView Convert_to_DataPoints(IEnumerable<GistureData> input, int kind)
        {
            IEnumerable<DataPoint> Middle(IEnumerable<GistureData> input)
            {
                foreach (var item in input)
                {
                    yield return new DataPoint
                    {
                        Label = item.Kind == kind,
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
            return context.Data.LoadFromEnumerable(Middle(input));
        }
        public void Test()
        {
            var data2 = context.Data.LoadFromTextFile<GistureData>("./csvs/persion1.csv", hasHeader: false, separatorChar: ',');//IdataView?
            var middle_data2 = context.Data
                .CreateEnumerable<GistureData>(data2, reuseRowObject: false);
            var final_data2 = Convert_to_DataPoints(input: middle_data2, kind: 1);

            var transform_predicts = models[0].Transform(final_data2);
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
        private class Prediction
        {
            public bool Label { get; set; }
            public bool PredictedLabel { get; set; }
        }

    }

}
