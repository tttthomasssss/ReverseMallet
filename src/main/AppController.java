package main;

import cc.mallet.classify.*;
import cc.mallet.classify.constraints.ge.MaxEntFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntGEConstraint;
import cc.mallet.classify.constraints.ge.MaxEntKLFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntL2FLGEConstraints;
import cc.mallet.optimize.ConjugateGradient;
import cc.mallet.optimize.GradientAscent;
import cc.mallet.optimize.Optimizer;
import cc.mallet.optimize.OrthantWiseLimitedMemoryBFGS;
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

        /**
         * Lots of different MaxEntGETrainer trials...
         */
        ArrayList<MaxEntGEConstraint> constraints = this.createConstraints(trainingData, BASE_DATA_FOLDER + BASEBALL_HOCKEY_CONSTRAINTS_FILE, false, false, false);
        MaxEntGETrainer meTrainer = new MaxEntGETrainer(constraints);

        // Create Optimizer, needs an Optimizable as Argument, see MaxEntGETrainer getOptimizable() for what it does and takes, in essence it is a http://mallet.cs.umass.edu/api/cc/mallet/classify/MaxEntOptimizableByGE.html instance
        Optimizer optimizer = new ConjugateGradient(meTrainer.getOptimizable(trainingData));
        //Optimizer optimizer = new GradientAscent(meTrainer.getOptimizable(trainingData));
        //Optimizer optimizer = new OrthantWiseLimitedMemoryBFGS(meTrainer.getOptimizable(trainingData)); // see http://research.microsoft.com/en-us/um/people/jfgao/paper/icml07scalable.pdf

        meTrainer.setOptimizer(optimizer);

        // TODO: Find out whether or not the optimizer needs to be reset before training
        meModel = meTrainer.train(trainingData);

        factory.putClassifier("maxEntGe-conjugateGradient", meModel);



        // ...AND FOR OTHER REASONS NORMAL MAXENT NEEDS LABELED TRAINING DATA

        //allData = this.splitData(Util.readZipData(BASE_DATA_FOLDER + BASEBALL_HOCKEY_FILE, Util.getPipe("document"), null), false);
        //trainingData = allData[0];

        //factory.setupMaxEntClassifier(trainingData, 1);


        // Print Metrics
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

    private ArrayList<MaxEntGEConstraint> createConstraints(InstanceList trainingData, String constraintsFile, boolean useValues, boolean l2, boolean normalize)
    {
        /**
         * In order to use a different Optimizer, we need to create the constraints ourselves
         * The default implementation of MaxEntGETrainer simply creates LimitedMemoryBFGS Optimizer
         * directly in the train method (which is a bit annoying).
         * As the constraints are also created in the train method, we first have to create the constraints
         * and then create an Optimizer.
         *
         * In case L2 penalty is enabled (see http://people.cs.umass.edu/~mccallum/papers/druck08sigir.pdf & http://mallet.cs.umass.edu/ge-classification.php for more info)
         * use MaxEntL2FLGEConstraints: http://mallet.cs.umass.edu/api/cc/mallet/classify/constraints/ge/MaxEntKLFLGEConstraints.html
         *      Constructor: numFeatures, numLabels, useValues, normalize
         *
         * otherwise use MaxEntKLFLGEConstraints: http://mallet.cs.umass.edu/api/cc/mallet/classify/constraints/ge/MaxEntKLFLGEConstraints.html
         *      Constructor: numFeatures, numLabels, useValues
         */

        // Read the Constraints from File
        HashMap<Integer, double[]> constraintsMap = FeatureConstraintUtil.readConstraintsFromFile(constraintsFile, trainingData);

        // Initialise Constraints
        ArrayList<MaxEntGEConstraint> constraints = new ArrayList<MaxEntGEConstraint>();
        MaxEntFLGEConstraints geConstraints = null;

        // Create Constraints
        if (l2) {
            geConstraints = new MaxEntL2FLGEConstraints(trainingData.getDataAlphabet().size(), trainingData.getTargetAlphabet().size(), useValues, normalize);
            for (int fi : constraintsMap.keySet()) {
                geConstraints.addConstraint(fi, constraintsMap.get(fi), 1);
            }
            constraints.add(geConstraints);
        } else {
            geConstraints = new MaxEntKLFLGEConstraints(trainingData.getDataAlphabet().size(), trainingData.getTargetAlphabet().size(), useValues);
            for (int fi : constraintsMap.keySet()) {
                geConstraints.addConstraint(fi, constraintsMap.get(fi), 1);
            }
        }

        constraints.add(geConstraints);

        return constraints;
    }

    public void stop(int exitCode)
    {
        System.exit(exitCode);
    }
}
