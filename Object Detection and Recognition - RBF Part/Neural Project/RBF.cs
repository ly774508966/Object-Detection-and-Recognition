using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Project
{
    class RBF
    {
        const int classes = 5;
        const int features = 128;
        const int num_test_images = 14;
        const int frame_size = 4;

        const int Cat = 1;
        const int Laptop = 2;
        const int Apple = 3;
        const int Car = 4;
        const int Helicopter = 5;

        List<Matrix<double>>[] training_set, frames_set, clusters;
        TestImage[] images;
        int sumTraining, sumTesting, num_of_clusters, epochs, threshold;
        Matrix<double> mean, maximum;
        Matrix<double>[] centroids;
        double[,] weights;
        double eta;
        double[] variances, phi;
        int[,] confusion_matrix;
        DataGridView dataGridView;

        public RBF(double eta, int epochs, int num_of_clusters, int threshold, ref DataGridView dataGridView)
        {
            this.eta = eta;
            this.epochs = epochs;
            this.num_of_clusters = num_of_clusters;
            this.threshold = threshold;
            this.dataGridView = dataGridView;
            training_set = new List<Matrix<double>>[classes + 1];
            //testing_set = new List<Matrix<double>>[classes + 1];
            images = new TestImage[num_test_images];
            frames_set = new List<Matrix<double>>[classes + 1];
            centroids = new Matrix<double>[num_of_clusters + 1];
            weights = new double[classes + 1, num_of_clusters + 1];
            clusters = new List<Matrix<double>>[num_of_clusters + 1];
            variances = new double[num_of_clusters + 1];
            phi = new double[num_of_clusters + 1];
            confusion_matrix = new int[classes + 2, classes + 2];

            for (int i = 1; i <= classes; i++)
            {
                training_set[i] = new List<Matrix<double>>();
                //testing_set[i] = new List<Matrix<double>>();
                frames_set[i] = new List<Matrix<double>>();
            }

            for (int i = 0; i < 14; i++)
            {
                images[i] = new TestImage();
            }

            for (int i = 0; i <= num_of_clusters; i++)
            {
                clusters[i] = new List<Matrix<double>>();
            }
        }

        public void displayConfusionMatrix()
        {
            dataGridView.Rows.Clear();
            for (int i = 1; i < classes + 2; i++)
            {
                var row = new DataGridViewRow();
                for (int j = 1; j < classes + 2; j++)
                {
                    row.Cells.Add(new DataGridViewTextBoxCell()
                    {
                        Value = confusion_matrix[i, j]
                    });
                }
                dataGridView.Rows.Add(row);
            }
        }

        public void prepareConfusionMatrix()
        {
            for (int i = 1; i < classes + 2; i++)
            {
                for (int j = 1; j < classes + 1; j++)
                {
                    confusion_matrix[i, classes + 1] += confusion_matrix[i, j];
                }
            }

            for (int i = 1; i < 5; i++)
            {
                for (int j = 1; j < 4; j++)
                {
                    confusion_matrix[classes + 1, i] += confusion_matrix[j, i];
                }
            }
        }

        public string calculateOverallAccuracy()
        {
            int sumDiagonal = 0;
            for (int i = 1; i <= classes; i++)
            {
                sumDiagonal += confusion_matrix[i, i];
            }

            //int sumTesting = 60;
            double overallAccuracy = (double)sumDiagonal / 24;
            overallAccuracy *= 100;
            overallAccuracy = Math.Round(overallAccuracy, 2);
            return overallAccuracy.ToString() + "%";
        }

        public void classify()
        {

        }

        public void updateWeights(double error, int class_num)
        {
            //1st method
            for (int i = 1; i <= num_of_clusters; i++)
            {
                weights[class_num, i] += eta * error * phi[i];
            }
        }

        public void startTesting()
        {
            for (int image_num = 0; image_num < num_test_images; image_num++)
            {
                int[] cnt = new int[classes + 1];

                for (int i = 0; i < images[image_num].testing_set.Count; i++)
                {
                    for (int j = 1; j <= num_of_clusters; j++)
                    {
                        double r = euclideanDistance(images[image_num].testing_set[i], centroids[j]);
                        phi[j] = Math.Exp((-r * r) / (2 * variances[j]));
                    }

                    double max = -1e9;
                    int y = 0;
                    for (int j = 1; j <= classes; j++)
                    {
                        double sum = 0;
                        for (int k = 1; k <= num_of_clusters; k++)
                        {
                            sum += phi[k] * weights[j, k];
                        }
                        if (sum > max)
                        {
                            max = sum;
                            y = j;
                        }
                    }

                    cnt[y]++;
                    images[image_num].frames[y].Add(images[image_num].temp_frames[i]);
                }

                List<int> actual = new List<int>();

                for (int i = 1; i <= classes; i++)
                {
                    if (cnt[i] > threshold)
                    {
                        actual.Add(i);
                    }
                }

                bool[] vis = new bool[classes + 1];

                for (int i = 0; i < actual.Count; i++)
                {
                    bool found = false;

                    for (int j = 0; j < images[image_num].labels.Count; j++)
                    {
                        if (actual[i] == images[image_num].labels[j])
                        {
                            found = true;
                            vis[actual[i]] = true;
                            break;
                        }
                    }

                    if (found)
                        confusion_matrix[actual[i], actual[i]]++;
                    else
                    {
                        bool f = false;

                        for (int j = 1; j <= classes; j++)
                        {
                            if (!vis[j])
                            {
                                f = true;
                                confusion_matrix[actual[i], j]++;
                                break;
                            }
                        }

                        if (!f)
                            confusion_matrix[actual[i], 1]++;
                    }
                }
            }
        }

        public void startTraining()
        {
            for (int e = 0; e < epochs; e++)
            {
                for (int class_num = 1; class_num <= classes; class_num++)
                {
                    for (int i = 0; i < training_set[class_num].Count; i++)
                    {
                        for (int j = 1; j <= num_of_clusters; j++)
                        {
                            double r = euclideanDistance(training_set[class_num][i], centroids[j]);
                            phi[j] = Math.Exp((-r * r) / (2 * variances[j]));
                        }

                        for (int j = 1; j <= classes; j++)
                        {
                            double y = 0;

                            for (int k = 1; k <= num_of_clusters; k++)
                            {
                                y += weights[j, k] * phi[k];
                            }

                            double d = (class_num == j) ? 1 : 0;
                            double error = d - y;
                            updateWeights(error, j);
                        }

                        //2nd method
                        //double[] d = new double[classes];
                        //double[] y = new double[classes];

                        //d[class_num - 1] = 1;

                        //for (int j = 1; j <= classes; j++)
                        //{
                        //    double sum = 0;

                        //    for (int k = 1; k <= num_of_clusters; k++)
                        //    {
                        //        sum += weights[j, k] * phi[k];
                        //    }

                        //    y[j - 1] = sum;
                        //}

                        //double[] error = new double[classes];
                        //for (int j = 0; j < classes; j++)
                        //{
                        //    error[j] = d[j] - y[j];
                        //}

                        //updateWeights(error, class_num);
                    }
                }
            }
        }

        //public double startTesting()
        //{
        //    //for (int i = 0; i < testing_set[class_num].Count; i++)
        //    //{
        //    //    for (int j = 0; j < 4; j++)
        //    //    {
        //    //        testing_set[class_num][i][j, 0] -= mean[j, 0];
        //    //        max[j, 0] = Math.Max(max[j, 0], testing_set[class_num][i][j, 0]);
        //    //    }
        //    //}
        //    int correct_samples = 0;

        //    for (int class_num = 1; class_num <= classes; class_num++)
        //    {
        //        for (int i = 0; i < testing_set[class_num].Count; i++)
        //        {
        //            for (int j = 1; j <= num_of_clusters; j++)
        //            {
        //                for (int k = 0; k < features; k++)
        //                {
        //                    testing_set[class_num][i][k, 0] -= mean[k, 0];
        //                    testing_set[class_num][i][k, 0] /= maximum[k, 0];
        //                }

        //                double r = euclideanDistance(testing_set[class_num][i], centroids[j]);
        //                phi[j] = Math.Exp((-r * r) / (2 * variances[j]));
        //            }

        //            double max = -1e9;
        //            int y = 0;
        //            for (int j = 1; j <= classes; j++)
        //            {
        //                double sum = 0;
        //                for (int k = 1; k <= num_of_clusters; k++)
        //                {
        //                    sum += phi[k] * weights[j, k];
        //                }
        //                if (sum > max)
        //                {
        //                    max = sum;
        //                    y = j;
        //                }
        //            }

        //            cnt[y]++;
        //            sumTesting++;
        //        }
        //    }
        //    double accuracy = (double)correct_samples / sumTesting;
        //    accuracy *= 100;
        //    return Math.Round(accuracy, 2);
        //}

        public void calculateVarainces()
        {
            for (int c = 1; c <= num_of_clusters; c++)
            {
                double sum = 0;

                for (int i = 0; i < clusters[c].Count; i++)
                    sum += euclideanDistance(clusters[c][i], centroids[c]);

                variances[c] = sum / clusters[c].Count;
            }
        }

        public double euclideanDistance(Matrix<double> sample, Matrix<double> centroid)
        {
            double dist = 0;
            for (int i = 0; i < features; i++)
                dist += Math.Pow(sample[i, 0] - centroid[i, 0], 2);
            dist = Math.Sqrt(dist);
            return dist;
        }

        public void k_means()
        {
            int num_of_iterations = 0;
            while (num_of_iterations < 100)
            {
                for (int i = 1; i <= num_of_clusters; i++)
                    clusters[i].Clear();

                for (int class_num = 1; class_num <= classes; class_num++)
                {
                    for (int i = 0; i < training_set[class_num].Count; i++)
                    {
                        double min = 1e9;
                        int sample_cluster = 0;
                        for (int j = 1; j <= num_of_clusters; j++)
                        {
                            double dist = euclideanDistance(training_set[class_num][i], centroids[j]);
                            if (dist < min)
                            {
                                min = dist;
                                sample_cluster = j;
                            }
                        }

                        clusters[sample_cluster].Add(training_set[class_num][i]);
                    }
                }

                bool flag = false;

                for (int cluster_num = 1; cluster_num <= num_of_clusters; cluster_num++)
                {
                    Matrix<double> avg = Matrix<double>.Build.Dense(features, 1);

                    for (int i = 0; i < clusters[cluster_num].Count; i++)
                    {
                        for (int j = 0; j < features; j++)
                        {
                            avg[j, 0] += clusters[cluster_num][i][j, 0];
                        }
                    }

                    for (int i = 0; i < features; i++)
                        avg[i, 0] /= clusters[cluster_num].Count;

                    //If any centroid at the current iteration differs from the centroid of the previous one, then we continue k-means algorithm
                    //If the centroids of the current iteration are identical to the previous one, then we stop k-means algorithm
                    //if (mean != centroids[cluster_num])
                    //    flag = true;

                    for (int i = 0; i < features; i++)
                    {
                        if (avg[i, 0] != centroids[cluster_num][i, 0])
                        {
                            flag = true;
                            break;
                        }
                    }

                    //centroids[cluster_num] = avg;
                    for (int i = 0; i < features; i++)
                        centroids[cluster_num][i, 0] = avg[i, 0];
                }
                if (!flag)
                    break;
                num_of_iterations++;
            }
        }

        public void initWeights()
        {
            Random rnd = new Random();

            for (int i = 0; i < classes; i++)
            {
                for (int j = 0; j <= num_of_clusters; j++)
                {
                    weights[i, j] = rnd.NextDouble();
                }
            }
        }

        public void initCentroids()
        {
            Random rnd = new Random();
            HashSet<Tuple<int, int>> vis = new HashSet<Tuple<int, int>>();

            for (int i = 1; i <= num_of_clusters; i++)
            {
                int class_num, sample_num;
                while (true)
                {
                    class_num = rnd.Next(1, classes + 1);
                    sample_num = rnd.Next(training_set[class_num].Count);
                    Tuple<int, int> t = new Tuple<int,int>(class_num, sample_num);
                    if (!vis.Contains(t))
                    {
                        vis.Add(t);
                        centroids[i] = training_set[class_num][sample_num];
                        break;
                    }
                }
            }
        }

        public int mapClass(string token)
        {
            if (token == "Cat")
                return 1;
            if (token == "Laptop")
                return 2;
            if (token == "Apple")
                return 3;
            if (token == "Car")
                return 4;
            return 5;   //Helicopter
        }

        public void readTrainingSet()
        {
            StreamReader labels = new StreamReader(@"E:\College\Neural Network\Labs\Project\label_name.txt");
            StreamReader descriptors = new StreamReader(@"E:\College\Neural Network\Labs\Project\descriptors.txt");
            StreamReader frames = new StreamReader(@"E:\College\Neural Network\Labs\Project\frame.txt");

            string line_labels, line_descriptors, line_frames;

            while ((line_labels = labels.ReadLine()) != null)
            {
                line_descriptors = descriptors.ReadLine();
                line_frames = frames.ReadLine();

                int sample_label = mapClass(line_labels);

                string[] splitter = line_descriptors.Split(' ');
                int num_descriptors = int.Parse(splitter[0]);
                sumTraining += num_descriptors;

                for (int i = 0; i < num_descriptors; i++)
                {
                    //Read descriptor
                    line_descriptors = descriptors.ReadLine();
                    splitter = line_descriptors.Split(' ');

                    Matrix<double> sample_descriptor = Matrix<double>.Build.Dense(features, 1);

                    for (int j = 0; j < features; j++)
                    {
                        sample_descriptor[j, 0] = double.Parse(splitter[j]);
                    }

                    //Read frame
                    line_frames = frames.ReadLine();
                    splitter = line_frames.Split(' ');

                    Matrix<double> sample_frame = Matrix<double>.Build.Dense(frame_size, 1);

                    for (int j = 0; j < frame_size; j++)
                    {
                        sample_frame[j, 0] = double.Parse(splitter[j]);
                    }

                    //Insertion
                    training_set[sample_label].Add(sample_descriptor);
                    frames_set[sample_label].Add(sample_frame);
                }
            }
        }

        public void readTestingSet()
        {
            StreamReader labels = new StreamReader(@"E:\College\Neural Network\Labs\Project\testing_image_name.txt");
            StreamReader descriptors = new StreamReader(@"E:\College\Neural Network\Labs\Project\testing_descriptors.txt");
            StreamReader keypoints = new StreamReader(@"E:\College\Neural Network\Labs\Project\testing_keypoints_number.txt");
            StreamReader frames = new StreamReader(@"E:\College\Neural Network\Labs\Project\testing_frame.txt");

            List<string> images_list = Directory.GetFiles(@"E:\College\Neural Network\Labs\Project\Data set\Testing", "*.*", SearchOption.AllDirectories)
                .ToList();

            for (int i = 0; i < num_test_images; i++)
            {
                string line_labels = labels.ReadLine();
                string[] splitter = line_labels.Split(',');

                for (int j = 0; j < splitter.Length; j++)
                {
                    images[i].labels.Add(mapClass(splitter[j]));
                }

                string line_keypoints = keypoints.ReadLine();
                int num_descriptors = int.Parse(line_keypoints);

                for (int j = 0; j < num_descriptors; j++)
                {
                    string line_descriptors = descriptors.ReadLine();
                    splitter = line_descriptors.Split(',');

                    Matrix<double> m = Matrix<double>.Build.Dense(features, 1);

                    for (int k = 0; k < features; k++)
                    {
                        m[k, 0] = double.Parse(splitter[k]);
                    }

                    string line_frames = frames.ReadLine();
                    splitter = line_frames.Split(',');

                    Matrix<double> n = Matrix<double>.Build.Dense(frame_size, 1);

                    for (int k = 0; k < frame_size; k++)
                    {
                        n[k, 0] = double.Parse(splitter[k]);
                    }

                    images[i].testing_set.Add(m);
                    images[i].temp_frames.Add(n);
                }
            }
        }
    }
}
