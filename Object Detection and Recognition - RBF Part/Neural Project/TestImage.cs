using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Project
{
    class TestImage
    {
        const int classes = 5;

        //public Bitmap img { get; set; }
        public List<int> labels { get; set; }
        public List<Matrix<double>> testing_set { get; set; }
        public List<Matrix<double>> temp_frames { get; set; } 
        public List<Matrix<double>>[] frames { get; set; }

        public TestImage()
        {
            labels = new List<int>();
            testing_set = new List<Matrix<double>>();
            temp_frames = new List<Matrix<double>>();
            frames = new List<Matrix<double>>[classes + 1];

            for (int i = 1; i <= classes; i++)
            {
                frames[i] = new List<Matrix<double>>();
            }
        }
    }
}
