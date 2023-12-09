import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class MyRandomForest {
    private RandomForest myRandomForest;
    public MyRandomForest() {
        this.myRandomForest=new RandomForest();

    }

    private Instances read_data(String path) throws Exception {
        // CSVLoader loader=new CSVLoader();
        // loader.setSource(new File(path));
        // Instances dataSet = loader.getDataSet();
        // dataSet.setClassIndex(dataSet.numAttributes()-1);
        // return dataSet;
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

    public void setParam_num_of_attribute(Integer num_of_attribute){
        if (num_of_attribute!=null)
            this.myRandomForest.setNumFeatures(num_of_attribute);
    }

    public void setParam_num_inside_tree(Integer num_inside_tree){
        if (num_inside_tree!=null)
            this.myRandomForest.setNumIterations(num_inside_tree);
    }

    public void setParam_max_depth(Integer max_depth){
        if (max_depth!=null)
            this.myRandomForest.setMaxDepth(max_depth);
    }

    public void setParam_random_seed(Integer random_seed){
        if (random_seed!=null)
            this.myRandomForest.setSeed(random_seed);
    }

    public void setParam_numExecutionSlots(Integer numExecutionSlots){
        if (numExecutionSlots==null){
            numExecutionSlots=0;
            // 默认并行
        }
        else{
            this.myRandomForest.setNumExecutionSlots(numExecutionSlots);
        }
            
    }


    public void fit(String train_path,String save_model_path) throws Exception {
        Instances data_train=this.read_data(train_path);
        this.myRandomForest.buildClassifier(data_train);//训练函数
        SerializationHelper.write(save_model_path,this.myRandomForest);
    }


    public double count_accuracy(RandomForest classifier,Instances test) throws Exception {
        int num = test.numInstances();
        int right=0;
        for(int i=0;i<num;i++){
            double class_pred = classifier.classifyInstance(test.instance(i));
            double class_true = test.instance(i).classValue();
            if(class_pred == class_true) {
                right++;
            }
//            double[] prob_distribute = classifier.distributionForInstance(test.instance(i));
//            System.out.println("");
        }
        return (double)right/num;
    }

    public double predict_proba(String test_path, String output_path, String read_model_path) throws Exception {
        Instances test_train=this.read_data(test_path);
        RandomForest classifier = (RandomForest) SerializationHelper.read(read_model_path);
        double[][] test_proba = classifier.distributionsForInstances(test_train);//预测函数
        this.save_data(output_path,test_proba);
        return count_accuracy(classifier,test_train);
    }


}
