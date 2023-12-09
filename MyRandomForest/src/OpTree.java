import weka.classifiers.trees.OptimizedForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class OpTree {
    private OptimizedForest optimizedForest;

    public OpTree() {
        this.optimizedForest=new OptimizedForest();

    }

    private Instances read_data(String path) throws Exception {
        Instances dataSet = ConverterUtils.DataSource.read(path);
        dataSet.setClassIndex(dataSet.numAttributes()-1);
        return dataSet;
    }

    private void save_data(String path,double[][] data) throws IOException {
        File file=new File(path);
        FileWriter fw = new FileWriter(file);
        BufferedWriter bw=new BufferedWriter(fw);
        for(int i=0;i<data.length;i++) {
            for (int j = 0; j < data[0].length; j++) {
                bw.write(String.valueOf(data[i][j]));
                if (j!=(data[0].length-1))
                    bw.write(",");
            }
            bw.newLine();
        }
        bw.close();
        fw.close();
    }

    public void setParam_Iterations(Integer iterlation){
        if (iterlation!=null)
            this.optimizedForest.setNumberIterations(iterlation);
    }

    public void setParam_population(Integer population){
        if (population!=null)
        this.optimizedForest.setSizeOfPopulation(population);
    }


    public void fit(String train_path) throws Exception {
        Instances data_train=this.read_data(train_path);
        this.optimizedForest.buildClassifier(data_train);//训练函数
    }

    public double count_accuracy( Instances test) throws Exception {
        int num = test.numInstances();
        int right=0;
        for(int i=0;i<num;i++){
            double class_pred = this.optimizedForest.classifyInstance(test.instance(i));
            double class_true = test.instance(i).classValue();
            if(class_pred == class_true) {
                right++;
            }
//            double[] prob_distribute = classifier.distributionForInstance(test.instance(i));
//            System.out.println("");
        }
        return (double)right/num;
    }

    public double predict_proba(String test_path, String output_path) throws Exception {
        Instances test_data=this.read_data(test_path);
        double[][] test_proba = this.optimizedForest.distributionsForInstances(test_data);//预测函数
        this.save_data(output_path,test_proba);
        return count_accuracy(test_data);

    }



}
