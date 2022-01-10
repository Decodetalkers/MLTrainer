namespace Gisture
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainer = new Trainers();
            //trainer.Test();
			var forpredict = new DataPoint{
				ThumbAngle =  37.8417f,
				IndexAngle = 62.4674f,
			    MiddleAngle = -1.58702f,
				RingAngle = -20.0989f,
				PinkyAngle = -30.4727f,
				ThumbIndex = -59.3171f,
				IndexMiddle = 1.09537f,
			};
			Console.WriteLine(trainer.Predict(forpredict));

        }
    }
}
