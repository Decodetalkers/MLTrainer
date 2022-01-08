// See https://aka.ms/new-console-template for more info
using Microsoft.ML;
using Microsoft.ML.Data;
//using Microsoft.ML.Trainers;
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
            var data = context.Data.LoadFromTextFile<GistureData>("./csvs/persion4.csv", hasHeader: false, separatorChar: ',');//IdataView?

            var middle_data = context.Data
                .CreateEnumerable<GistureData>(data, reuseRowObject: false);

			var final_data = Convert_to_DataPoints(middle_data);
            var finnal_data = context.Data.LoadFromEnumerable(final_data);
            var pipeline = context.Transforms.Concatenate("Features", new[] { "ThumbIndex", "IndexAngle", "MiddleAngle", "RingAngle", "PinkyAngle", "ThumbIndex", "IndexMiddle" })
                .Append(context.Regression.Trainers.Sdca(labelColumnName: "Label", maximumNumberOfIterations: 10));
            var model = pipeline.Fit(data);
			var eight = new GistureData() {
				ThumbAngle=38.1197f,
				IndexAngle=90.39656f,
				MiddleAngle=-26.1411f,
				RingAngle=-35.6065f,
				PinkyAngle=-32.2617f,
				ThumbIndex=-71.2519f,
				IndexMiddle=-0.614772f,
			};
			var kind = context.Model.CreatePredictionEngine<GistureData,Prediction>(model).Predict(eight);
			Console.WriteLine(kind.Kind);

        }
		public class Prediction
		{
			[ColumnName("Score")]
			public float Kind {get;set;}
		}
        private static IEnumerable<DataPoint> Convert_to_DataPoints(IEnumerable<GistureData> input)

        {
            foreach (var item in input)
            {
                yield return new DataPoint
                {
                    Label = item.Kind,
                    Features = new float[7] {
                        item.RingAngle,
                        item.IndexAngle,
                        item.MiddleAngle,
                        item.RingAngle,
                        item.PinkyAngle,
                        item.ThumbIndex,
                        item.IndexMiddle
                    },

                };
            }
            //var random = new Random(seed);
            //int randomInt() => (int)random.NextInt64();
            //for (int i = 0; i < count; i++)
            //{
            //    var label = randomInt() ;
            //    yield return new DataPoint
            //    {
            //        Label = label,
            //        // Create random features that are correlated with the label.
            //        // For data points with false label, the feature values are
            //        // slightly increased by adding a constant.
            //        Features = Enumerable.Repeat(label, 50)
            //            .Select(x => x ? randomFloat() : randomFloat() +
            //            0.1f).ToArray()

            //    };
            //}
        }
    }

}
