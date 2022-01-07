// See https://aka.ms/new-console-template for more info
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Linq;

namespace Gisture
{
	class Program
	{
		static void Main(string[] args)
		{
			var context = new MLContext(); // MLContext?
			var data = context.Data.LoadFromTextFile<GistureData>("./csvs/persion1.csv",hasHeader:false,separatorChar:',');//IdataView?
			var options = new LinearSvmTrainer.Options {
				BatchSize = 10,
				PerformProjection = true,
				NumberOfIterations = 10
			}; //options?
			var pipeline = context.BinaryClassification.Trainers
				.LinearSvm(options);
		}
	}
    
}
