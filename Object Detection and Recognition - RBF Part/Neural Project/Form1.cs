using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Project
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            double eta = double.Parse(textBox1.Text);
            int epochs = int.Parse(textBox2.Text);
            int num_of_clusters = int.Parse(textBox3.Text);
            int threshold = int.Parse(textBox4.Text);

            RBF rbf = new RBF(eta, epochs, num_of_clusters, threshold, ref dataGridView1);

            //rbf.readData();
            //rbf.initCentroids();
            //rbf.initWeights();
            //rbf.k_means();
            //rbf.calculateVarainces();
            //rbf.startTraining();
            //rbf.startTesting();
            //rbf.prepareConfusionMatrix();
            //rbf.displayConfusionMatrix();
            //textBox4.Text = rbf.calculateOverallAccuracy();

            rbf.readTrainingSet();
            rbf.readTestingSet();
            rbf.initCentroids();
            rbf.initWeights();
            rbf.k_means();
            rbf.calculateVarainces();
            rbf.startTraining();
            rbf.startTesting();
            rbf.prepareConfusionMatrix();
            rbf.displayConfusionMatrix();
            textBox5.Text = rbf.calculateOverallAccuracy();

            MessageBox.Show("Done!");
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            textBox1.Text = "0.9";
            textBox2.Text = "50";
            textBox3.Text = "2";
            textBox4.Text = "3";
        }
    }
}
