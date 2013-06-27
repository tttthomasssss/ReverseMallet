package main;

import cc.mallet.classify.*;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Multinomial;
import dualist.classify.NaiveBayesWithPriorsTrainer;
import dualist.tui.Util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: thk22
 * Date: 25/06/2013
 * Time: 16:36
 * To change this template use File | Settings | File Templates.
 */
public class AppController {

    private static final String BASE_DATA_FOLDER = "/Volumes/LocalDataHD/thk22/DevSandbox/_data/MoreMalletTests/sample-data/";
    private static final String BASEBALL_HOCKEY_FILE = "baseball-hockey.zip";
    private static final String BASEBALL_HOCKEY_CONSTRAINTS_FILE = "sport/sport1.constraints";
    private static final String BASEBALL_HOCKEY_RANGE_CONSTRAINTS_FILE = "sport/sport_range1.constraints";

    public static void main(String[] args)
    {
        System.out.println("Starting...");

        AppController app = new AppController();

        app.start();
    }

    private void start()
    {
        MalletFactory factory = new MalletFactory();

        // MAXENT GE NEEDS UNLABELED TRAINING DATA...
        InstanceList[] allData = this.splitData(Util.readZipData(BASE_DATA_FOLDER + BASEBALL_HOCKEY_FILE, Util.getPipe("document"), null), true);

        InstanceList trainingData = allData[0];
        InstanceList testingData = allData[1];

        MaxEnt meModel = factory.setupMaxEntGEClassifier(trainingData, BASE_DATA_FOLDER + BASEBALL_HOCKEY_CONSTRAINTS_FILE, 1);

        // MaxEntGERangeTrainer
        //factory.setupMaxEntGERangeClassifier(trainingData, BASE_DATA_FOLDER + BASEBALL_HOCKEY_RANGE_CONSTRAINTS_FILE, 1);



        // ...AND FOR OTHER REASONS NORMAL MAXENT NEEDS LABELED TRAINING DATA

        //allData = this.splitData(Util.readZipData(BASE_DATA_FOLDER + BASEBALL_HOCKEY_FILE, Util.getPipe("document"), null), false);
        //trainingData = allData[0];

        //factory.setupMaxEntClassifier(trainingData, 1);



        HashMap<String, Double> metrics = factory.classifyAll(testingData);

        for (String key : metrics.keySet()) {
            System.out.println(key + ": " + metrics.get(key));
        }
    }

    public InstanceList[] splitData(InstanceList allInstances, boolean hideTrainingDataLabels)
    {
        InstanceList[] trainingAndTestingData = new InstanceList[2];

        InstanceList trainingData = new InstanceList(Util.getPipe("document"));
        InstanceList testingData = new InstanceList(Util.getPipe("document"));

        Instance trainingInstance = null;

        for (int i = 0; i < allInstances.size(); i++) {
            if (i % 2 == 0 || i % 3 == 0 || i % 7 == 0) { // Training data, Java's random is not random enough...

                trainingInstance = allInstances.get(i);

                // Mallet GE doesn't like labeled training data, so hide it
                // Check MaxEntOptimizableByGE lines 152 and 175
                if (hideTrainingDataLabels) {
                    trainingInstance.unLock();
                    trainingInstance.setTarget(null);
                    trainingInstance.lock();
                }

                trainingData.add(allInstances.get(i));
            } else { // Testing Data
                testingData.add(allInstances.get(i));
            }
        }

        trainingData.getPipe().setDataAlphabet(allInstances.getDataAlphabet());
        trainingData.getPipe().setTargetAlphabet(allInstances.getTargetAlphabet());
        testingData.getPipe().setDataAlphabet(allInstances.getDataAlphabet());
        testingData.getPipe().setTargetAlphabet(allInstances.getTargetAlphabet());

        trainingAndTestingData[0] = trainingData;
        trainingAndTestingData[1] = testingData;

        return trainingAndTestingData;
    }

    public void stop(int exitCode)
    {
        System.exit(exitCode);
    }
}
