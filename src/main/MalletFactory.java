package main;

import cc.mallet.classify.*;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import dualist.tui.Util;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created with IntelliJ IDEA.
 * User: thk22
 * Date: 26/06/2013
 * Time: 09:23
 * To change this template use File | Settings | File Templates.
 */
public class MalletFactory {

    private static final String ACCURACY_KEY = "Accuracy-";
    private static final String PRECISION_KEY = "Precision-";
    private static final String RECALL_KEY = "Recall-";
    private static final String F_SCORE_KEY = "F-Score-";
    private static final String MAXENT_GE_CLASSIFIER_KEY = "maxEntGE";
    private static final String MAXENT_GE_RANGE_CLASSIFIER = "maxEntGERange";
    private static final String MAXENT_CLASSIFIER_KEY = "maxEnt";

    private HashMap<String, Classifier> classifiers;

    public MalletFactory()
    {
        super();

        this.classifiers = new HashMap<String, Classifier>();
    }

    public void putClassifier(String key, Classifier c)
    {
        this.classifiers.put(key, c);
    }

    public MaxEnt setupMaxEntGERangeClassifier(InstanceList trainingData, String constraintsFile, double gaussianPriorVariance)
    {
        // Setup Trainer
        MaxEntGERangeTrainer meTrainer = new MaxEntGERangeTrainer();

        meTrainer.setConstraintsFile(constraintsFile);
        meTrainer.setGaussianPriorVariance(gaussianPriorVariance);

        // Train & Return Model
        MaxEnt meModel = meTrainer.train(trainingData);
        this.classifiers.put(MAXENT_GE_RANGE_CLASSIFIER, meModel);

        return meModel;
    }

    public MaxEnt setupMaxEntGEClassifier(InstanceList trainingData, String constraintsFile, double gaussianPriorVariance)
    {
        // Setup Trainer
        MaxEntGETrainer meTrainer = new MaxEntGETrainer();

        meTrainer.setConstraintsFile(constraintsFile);
        meTrainer.setGaussianPriorVariance(gaussianPriorVariance);
        meTrainer.setL2(false);

        // Train & Return Model
        MaxEnt meModel = meTrainer.train(trainingData);
        this.classifiers.put(MAXENT_GE_CLASSIFIER_KEY, meModel);

        return meModel;
    }

    public MaxEnt setupMaxEntClassifier(InstanceList trainingData, double gaussianPriorVariance)
    {
        // Setup Trainer
        MaxEntTrainer meTrainer = new MaxEntTrainer();

        meTrainer.setGaussianPriorVariance(gaussianPriorVariance);

        // Train & Return Model
        MaxEnt meModel = meTrainer.train(trainingData);
        this.classifiers.put(MAXENT_CLASSIFIER_KEY, meModel);

        return meModel;
    }

    public HashMap<String, Double> classifyAll(InstanceList testingData)
    {
        int n = testingData.size();

        // Accuracy
        int classifiedCorrectly = 0;

        HashMap<String, Double> metrics = new HashMap<String, Double>();
        Classification classification = null;
        Classifier c = null;

        for (String key : this.classifiers.keySet()) {

            c = this.classifiers.get(key);

            for (Instance i : testingData) {
                classification = c.classify(i);

                classifiedCorrectly += (classification.getLabeling().getBestLabel().toString().equals(i.getTarget().toString())) ? 1 : 0;
            }

            metrics.put(ACCURACY_KEY + key, (double)classifiedCorrectly / (double)n);
            classifiedCorrectly = 0;
        }

        return metrics;
    }
}
