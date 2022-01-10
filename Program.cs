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
			var forpredict2 = new DataPoint {
				ThumbAngle=62.4204f,
				IndexAngle= -8.53424f,
				MiddleAngle= 1.08902f,
				RingAngle=	157.382f,
				PinkyAngle= 110.563f,
				ThumbIndex =  60.698f,
				IndexMiddle =  5.03313f
			};
			Console.WriteLine(trainer.Predict(forpredict));
			Console.WriteLine(trainer.Predict(forpredict2));

        }
    }
}
