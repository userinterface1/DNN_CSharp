namespace Deep_Learning_DNN
{
    using System;
    using System.IO;

    internal class Program
    {
        static float[] targets1, targets2;
        static float[] inputs1, inputs2;
        static int batchSize;

        static float target1, target2;

        static float input1, input2;
        static float w1, w2, w3, w4;
        static float w5, w6, w7, w8;
        static float h1, h2;
        static float output1, output2;

        static float total;

        static float learning_rate;

        static int epoch;

        static void propagation(float input1, float input2, float target1, float target2)
        {
            float z1 = input1 * w1 + input2 * w2;
            h1 = sigmoid(z1);

            float z2 = input1 * w3 + input2 * w4;
            h2 = sigmoid(z2);

            float z3 = h1 * w5 + h2 * w6;
            output1 = sigmoid(z3);

            float z4 = h1 * w7 + h2 * w8;
            output2 = sigmoid(z4);

            float e1 = 1.0f / 2.0f * (target1 - output1) * (target1 - output1);
            float e2 = 1.0f / 2.0f * (target2 - output2) * (target2 - output2);

            total = e1 + e2;
        }

        static void backpropagation(float input1, float input2, float target1, float target2)
        {
            float totaldoutput1 = -(target1 - output1);
            float o1dz3 = output1 * (1 - output1);
            float z3dw5 = h1;
            float totaldw5 = totaldoutput1 * o1dz3 * z3dw5;

            float z3dw6 = h2;
            float totaldw6 = totaldoutput1 * o1dz3 * z3dw6;

            float totaldoutput2 = -(target2 - output2);
            float o2dz3 = output2 * (1 - output2);
            float z4dw7 = h1;
            float totaldw7 = totaldoutput2 * o2dz3 * z4dw7;

            float z4dw8 = h2;
            float totaldw8 = totaldoutput2 * o2dz3 * z4dw8;

            float e1doutput1 = -(target1 - output1);
            float output1dz3 = output1;
            float z3dh1 = w5;
            float e1dh1 = e1doutput1 * output1dz3 * z3dh1;

            float e2doutput2 = -(target2 - output2);
            float output2dz4 = output2;
            float z4dh1 = w7;
            float e2dh1 = e2doutput2 * output2dz4 * z4dh1;

            float totaldh1 = e1dh1 + e2dh1;

            float z3dh2 = h2;
            float e1dh2 = e1doutput1 * output1dz3 * z3dh2;

            float z4dh2 = h2;
            float e2dh2 = e2doutput2 * output2dz4 * z4dh2;

            float totaldh2 = e1dh2 + e2dh2;

            float h1dz1 = h1 * (1 - h1);
            float z1dw1 = input1;
            float totaldw1 = totaldh1 * h1dz1 * z1dw1;

            float z1dw2 = input2;
            float totaldw2 = totaldh1 * h1dz1 * z1dw2;

            float h2dz2 = h2 * (1 - h2);
            float z2dw3 = input1;
            float totaldw3 = totaldh2 * h2dz2 * z2dw3;

            float z2dw4 = input2;
            float totaldw4 = totaldh2 * h2dz2 * z2dw4;

            w1 -= learning_rate * totaldw1;
            w2 -= learning_rate * totaldw2;
            w3 -= learning_rate * totaldw3;
            w4 -= learning_rate * totaldw4;
            w5 -= learning_rate * totaldw5;
            w6 -= learning_rate * totaldw6;
            w7 -= learning_rate * totaldw7;
            w8 -= learning_rate * totaldw8;
        }

        static float sigmoid(float z)
        {
            return (float)(1 / (1 + Math.Exp(-z)));
        }

        static void display()
        {
            Console.WriteLine("목표 1 : " + output1 + " , 목표 2 : " + output2 + ", 오차값 : " + total);
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Learning? : ");
            string yorn = Console.ReadLine();
            if (yorn == "y")
            {
                Console.Write("Batch Size: ");
                batchSize = int.Parse(Console.ReadLine());

                targets1 = new float[batchSize];
                targets2 = new float[batchSize];
                inputs1 = new float[batchSize];
                inputs2 = new float[batchSize];

                for (int i = 0; i < batchSize; i++)
                {
                    Console.Write($"Target 1 for batch {i + 1}: ");
                    targets1[i] = float.Parse(Console.ReadLine());
                    Console.Write($"Target 2 for batch {i + 1}: ");
                    targets2[i] = float.Parse(Console.ReadLine());
                    Console.Write($"Input 1 for batch {i + 1}: ");
                    inputs1[i] = float.Parse(Console.ReadLine());
                    Console.Write($"Input 2 for batch {i + 1}: ");
                    inputs2[i] = float.Parse(Console.ReadLine());
                }

                Console.Write("Learning Rate : ");
                learning_rate = float.Parse(Console.ReadLine());

                Console.Write("Epoch : ");
                epoch = int.Parse(Console.ReadLine());

                Console.Write("Reset? : ");
                yorn = Console.ReadLine();

                Random rand = new Random();
                if (yorn == "y")
                {
                    w1 = (float)rand.NextDouble();
                    w2 = (float)rand.NextDouble();
                    w3 = (float)rand.NextDouble();
                    w4 = (float)rand.NextDouble();
                    w5 = (float)rand.NextDouble();
                    w6 = (float)rand.NextDouble();
                    w7 = (float)rand.NextDouble();
                    w8 = (float)rand.NextDouble();
                }
                else if (yorn == "n")
                {
                    load_weight();
                }

                for (int i = 0; i < epoch; ++i)
                {
                    for (int j = 0; j < batchSize; ++j)
                    {
                        propagation(inputs1[j], inputs2[j], targets1[j], targets2[j]);
                        display();
                        backpropagation(inputs1[j], inputs2[j], targets1[j], targets2[j]);
                    }
                }

                Console.WriteLine("W1 : " + w1);
                Console.WriteLine("W2 : " + w2);
                Console.WriteLine("W3 : " + w3);
                Console.WriteLine("W4 : " + w4);
                Console.WriteLine("W5 : " + w5);
                Console.WriteLine("W6 : " + w6);
                Console.WriteLine("W7 : " + w7);
                Console.WriteLine("W8 : " + w8);

                save_weight();
            }
            else if (yorn == "n")
            {
                load_weight();

                Console.Write("Input 1 : ");
                input1 = float.Parse(Console.ReadLine());
                Console.Write("Input 2 : ");
                input2 = float.Parse(Console.ReadLine());

                propagation(input1, input2, target1, target2);
                display();
            }
        }

        static void save_weight()
        {
            File.WriteAllText(@"C:\Users\USER\Desktop\weight.txt", w1 + "," + w2 + "," + w3 + "," + w4 + "," + w5 + "," + w6 + "," + w7 + "," + w8, System.Text.Encoding.UTF8);
        }

        static void load_weight()
        {
            string weight = File.ReadAllText(@"C:\Users\USER\Desktop\weight.txt", System.Text.Encoding.UTF8);
            string[] weights = weight.Split(',');
            w1 = float.Parse(weights[0]);
            w2 = float.Parse(weights[1]);
            w3 = float.Parse(weights[2]);
            w4 = float.Parse(weights[3]);
            w5 = float.Parse(weights[4]);
            w6 = float.Parse(weights[5]);
            w7 = float.Parse(weights[6]);
            w8 = float.Parse(weights[7]);
        }
    }
}
