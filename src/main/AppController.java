package main;

import cc.mallet.classify.*;
import cc.mallet.classify.constraints.ge.MaxEntFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntGEConstraint;
import cc.mallet.classify.constraints.ge.MaxEntKLFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntL2FLGEConstraints;
import cc.mallet.classify.tui.Vectors2FeatureConstraints;
import cc.mallet.optimize.ConjugateGradient;
import cc.mallet.optimize.GradientAscent;
import cc.mallet.optimize.Optimizer;
import cc.mallet.optimize.OrthantWiseLimitedMemoryBFGS;
import cc.mallet.pipe.*;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.*;
import cc.mallet.util.CommandOption;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;
import dualist.classify.NaiveBayesWithPriorsTrainer;
import dualist.classify.Queries;
import dualist.pipes.Labelize;
import dualist.tui.Util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;
import java.util.regex.Pattern;

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

    public static void main(String[] args) throws Exception
    {
        System.out.println("Starting...");

        AppController app = new AppController();

        //app.startMaxEntTrainingRuns();
        //app.startMaxEntGEUserProvidedLabelsRuns();
        app.startNaiveBayesWithPriorsTestRuns();
        //app.startLDATopicModelingRuns();
    }

    private void startLDATopicModelingRuns() throws Exception
    {
        // http://mallet.cs.umass.edu/topics-devel.php
        // Setup some default pipe
        ArrayList<Pipe> pipeList = new ArrayList<Pipe>();

        // Pipes: lowercase, tokenize, remove stopwords, map to features
        pipeList.add( new Input2CharSequence() );
        pipeList.add( new CharSequenceReplace(Pattern.compile("\\<.*?>"), ""));
        pipeList.add( new CharSequenceReplace(Pattern.compile("\\<[A-Za-z]+"), "") );
        pipeList.add( new CharSequenceReplace(Pattern.compile("[\\n\\r][\\s\\r\\n]*[\\n\\r]+"), "\n\n") );
        pipeList.add( new CharSequenceReplace(Pattern.compile("&(.*?);"), "") );
        pipeList.add( new CharSequenceReplace(Pattern.compile("[0-9]+"), "00") );
        pipeList.add( new CharSequenceLowercase() );
        pipeList.add( new CharSequence2TokenSequence(Pattern.compile("[\\p{L}\\p{Mn}]+")) );
        pipeList.add( new TokenSequenceRemoveStopwords() );
        pipeList.add( new TokenSequence2FeatureSequence() );
        pipeList.add( new Labelize() );

        // Setup Training Data
        InstanceList[] allData = this.splitData(Util.readZipData(BASE_DATA_FOLDER + BASEBALL_HOCKEY_FILE, new SerialPipes(pipeList), null), false);

        InstanceList trainingData = allData[0];
        InstanceList testingData = allData[1];

        System.out.println("TARGET ALPHABET: " + trainingData.getTargetAlphabet());
        System.out.println("PIPE TARGET ALPHABET: " + trainingData.getPipe().getTargetAlphabet());

        // I can only see 2 topics
        int numTopics = trainingData.getTargetAlphabet().size();
        ParallelTopicModel model = new ParallelTopicModel((LabelAlphabet)trainingData.getTargetAlphabet(), 0.01 * numTopics, 0.01);

        model.addInstances(trainingData);
        model.setNumThreads(2);
        model.setNumIterations(100);
        model.estimate();

        Object[][] topwords = model.getTopWords(25);

        for (int i = 0; i < topwords.length; i++) {
            for (int j = 0; j < topwords[i].length; j++) {
                System.out.println("TOPWORD: " + topwords[i][j]);
            }
        }
        System.out.println("MORE WORDS: \n" + model.displayTopWords(25, true));

        ArrayList<TreeSet<IDSorter>> sortedWords = model.getSortedWords();

        for (int i = 0; i < numTopics; i++) {
            TreeSet<IDSorter> treeSet = sortedWords.get(i);

            System.out.println("TOPWORDS FOR " + ((LabelAlphabet) trainingData.getTargetAlphabet()).lookupLabel(i) + "... ");

            Iterator<IDSorter> iterator = treeSet.iterator();

            int rank = 0;
            IDSorter curr = null;
            while (iterator.hasNext() && rank < 25) {
                curr = iterator.next();
                System.out.println("TOPWORD: " + trainingData.getDataAlphabet().lookupObject(curr.getID()) + " WITH WEIGHT: " + curr.getWeight());
                //curr.
                rank++;
            }
        }

    }

    private void startNaiveBayesWithPriorsTestRuns()
    {
        // Setup Training Data
        InstanceList[] allData = this.splitData(Util.readZipData(BASE_DATA_FOLDER + BASEBALL_HOCKEY_FILE, Util.getPipe("document"), null), false);

        InstanceList trainingData = allData[0];
        InstanceList testingData = allData[1];


        //-- Queries Interlude --//

        // Passive Querying
        HashMultimap<Integer, String> labeledFeatures = HashMultimap.create();
        System.out.println("COMMON FEATURES PER LABEL: " + Queries.commonFeaturesPerLabel(labeledFeatures, trainingData, 100));

        // Active Querying
        MalletFactory factory = new MalletFactory();
        MaxEnt meModel = factory.setupMaxEntGEClassifier(trainingData, BASE_DATA_FOLDER + BASEBALL_HOCKEY_CONSTRAINTS_FILE, 1.0, 1.0, false);
        Queries.queryFeaturesPerLabelMI(meModel, labeledFeatures, this.maxEntClassificationData(meModel, trainingData.cloneEmpty(), trainingData, false), 100);


        // InfoGain Interlude
        InfoGain ig = new InfoGain(trainingData);
        for (int i = 0; i < 10; i++) {
            System.out.println("### OBJECT AT RANK: " + ig.getObjectAtRank(i).toString());
            System.out.println("### INDEX AT RANK: " + ig.getIndexAtRank(i));
        }





        // Setup Trainer
        NaiveBayesWithPriorsTrainer nbTrainer = new NaiveBayesWithPriorsTrainer(trainingData.getPipe());

        nbTrainer.setPriorMultinomialEstimator(new Multinomial.MEstimator(5));
        nbTrainer.setAlpha(50);

        // Create a few labeled Features and add them to the trainer

        // Baseball features
        nbTrainer.addLabelFeature("rec.sport.baseball", "series");
        nbTrainer.addLabelFeature("rec.sport.baseball", "baseball");
        nbTrainer.addLabelFeature("rec.sport.baseball", "hitter");
        nbTrainer.addLabelFeature("rec.sport.baseball", "innings");
        nbTrainer.addLabelFeature("rec.sport.baseball", "pitcher");

        // Hockey features
        nbTrainer.addLabelFeature("rec.sport.hockey", "puck");
        nbTrainer.addLabelFeature("rec.sport.hockey", "bruins");
        nbTrainer.addLabelFeature("rec.sport.hockey", "nhl");
        nbTrainer.addLabelFeature("rec.sport.hockey", "bruins");
        nbTrainer.addLabelFeature("rec.sport.hockey", "gretzky");

        NaiveBayes nbModel;

        InstanceList labeledSet = trainingData.cloneEmpty();

        nbModel = nbTrainer.train( trainingData.cloneEmpty());
        InstanceList trainSet2 = Util.probabilisticData(nbModel, labeledSet, trainingData);
        nbModel = nbTrainer.train (trainSet2);
    }

    private void startMaxEntGEUserProvidedLabelsRuns() {
        // Setup Training Data
        InstanceList[] allData = this.splitData(Util.readZipData(BASE_DATA_FOLDER + BASEBALL_HOCKEY_FILE, Util.getPipe("document"), null), true);

        InstanceList trainingData = allData[0];
        InstanceList testingData = allData[1];

        // Reverse Engineer Vectors2FeatureConstraints   << DEBUG FROM HERE
        String[] args = new String[3];

        args[0] = "--output \"" + BASE_DATA_FOLDER + "sport/sport_generated.constraints\"";
        args[1] = "--features-file \"" + BASE_DATA_FOLDER + "sport/sport_labels.labeled_features\"";
        args[2] = "--targets heuristic";

        Vectors2FeatureConstraints v = new Vectors2FeatureConstraints();

        Class[] params = new Class[1];
        params[0] = InstanceList.class;
        Method m = null;

        try {
            m = v.getClass().getDeclaredMethod("doStuff", params);
        } catch (NoSuchMethodException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }

        CommandOption.process(v.getClass(), args);

        try {
            m.invoke(trainingData);
        } catch (IllegalAccessException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        } catch (InvocationTargetException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }

        //v.getClass()..doStuff(trainingData);

        //CommandOption.process(Vectors2FeatureConstraints.class, args);
    }

    private void startMaxEntTrainingRuns()
    {
        MalletFactory factory = new MalletFactory();

        // MAXENT GE NEEDS UNLABELED TRAINING DATA...
        InstanceList[] allData = this.splitData(Util.readZipData(BASE_DATA_FOLDER + BASEBALL_HOCKEY_FILE, Util.getPipe("document"), null), true);

        InstanceList trainingData = allData[0];
        InstanceList testingData = allData[1];

        // MaxEntGEConstraints
        ArrayList<MaxEntGEConstraint> constraints = this.createConstraints(trainingData, BASE_DATA_FOLDER + BASEBALL_HOCKEY_CONSTRAINTS_FILE, false, false, false); // seems to work better without L2 penalty and useValues and normalize

        // MaxEntGETrainer, LimitedMemoryBFGS optimizer
        MaxEnt meModel = factory.setupMaxEntGEClassifier(trainingData, BASE_DATA_FOLDER + BASEBALL_HOCKEY_CONSTRAINTS_FILE, 1.0, 1.0, false);

        // MaxEntGERangeTrainer
        //factory.setupMaxEntGERangeClassifier(trainingData, BASE_DATA_FOLDER + BASEBALL_HOCKEY_RANGE_CONSTRAINTS_FILE, 1);

        // MaxEntGETrainer, ConjugateGradient optimizer
        MaxEntGETrainer meTrainer = new MaxEntGETrainer(constraints);
        meTrainer.setGaussianPriorVariance(1.0);
        meTrainer.setTemperature(2.0);

        // Create Optimizer, needs an Optimizable as Argument, see MaxEntGETrainer getOptimizable() for what it does and takes, in essence it is a http://mallet.cs.umass.edu/api/cc/mallet/classify/MaxEntOptimizableByGE.html instance
        Optimizer optimizer = new ConjugateGradient(meTrainer.getOptimizable(trainingData));
        meTrainer.setOptimizer(optimizer);

        meModel = meTrainer.train(trainingData);
        factory.putClassifier("maxEntGe-conjugateGradient", meModel);

        // MaxEntGETrainer, GradientAscent Optimizer
        meTrainer = new MaxEntGETrainer(constraints);
        meTrainer.setGaussianPriorVariance(1.0);
        meTrainer.setTemperature(2.0);

        optimizer = new GradientAscent(meTrainer.getOptimizable(trainingData));
        meTrainer.setOptimizer(optimizer);

        meModel = meTrainer.train(trainingData);
        factory.putClassifier("maxEntGE-gradientAscent", meModel);

        // MaxEntGETrainer, OrthantWiseLimitedMemoryBFGS optimizer
        meTrainer = new MaxEntGETrainer(constraints);
        meTrainer.setGaussianPriorVariance(1.0);
        meTrainer.setTemperature(1.0);

        optimizer = new OrthantWiseLimitedMemoryBFGS(meTrainer.getOptimizable(trainingData)); // see http://research.microsoft.com/en-us/um/people/jfgao/paper/icml07scalable.pdf
        meTrainer.setOptimizer(optimizer);

        meModel = meTrainer.train(trainingData);
        factory.putClassifier("maxEntGE-orthantWiseLimitedMemoryBFGS", meModel);

        // Normal LimitedMemoryBFGS seems to work best

        // MaxEntPRTrainer Setups
        MaxEntPRTrainer prTrainer = new MaxEntPRTrainer();
        prTrainer.setConstraintsFile(BASE_DATA_FOLDER + BASEBALL_HOCKEY_CONSTRAINTS_FILE);
        prTrainer.setPGaussianPriorVariance(0.0001);
        prTrainer.setQGaussianPriorVariance(1000000);

        meModel = prTrainer.train(trainingData);
        factory.putClassifier("maxEntPR", meModel);



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

        for (int fi : constraintsMap.keySet()) {
            System.out.println("THE KEY INDEX: " + fi);
            System.out.println("THE KEY: " + trainingData.getDataAlphabet().lookupObject(fi));
            double[] x = constraintsMap.get(fi);
            for (int i = 0; i < x.length; i++) {
                System.out.println("PROB=[" + x[i] + "] FOR LABEL=[" + trainingData.getTargetAlphabet().lookupObject(i) + "]");
            }
        }

        // Initialise Constraints
        ArrayList<MaxEntGEConstraint> constraints = new ArrayList<MaxEntGEConstraint>();
        MaxEntFLGEConstraints geConstraints = null;

        // Create Constraints
        if (l2) {
            geConstraints = new MaxEntL2FLGEConstraints(trainingData.getDataAlphabet().size(), trainingData.getTargetAlphabet().size(), useValues, normalize);
            for (int fi : constraintsMap.keySet()) {
                geConstraints.addConstraint(fi, constraintsMap.get(fi), 1);
            }
        } else {
            geConstraints = new MaxEntKLFLGEConstraints(trainingData.getDataAlphabet().size(), trainingData.getTargetAlphabet().size(), useValues);
            for (int fi : constraintsMap.keySet()) {
                geConstraints.addConstraint(fi, constraintsMap.get(fi), 1);
            }
        }

        constraints.add(geConstraints);

        return constraints;
    }

    public InstanceList maxEntClassificationData(MaxEnt meModel, InstanceList labeledSet, InstanceList unlabeledSet, boolean useMax)
    {

        InstanceList ret = labeledSet.shallowClone();
        for (Instance inst : unlabeledSet) {

            Instance inst2 = inst.shallowCopy();
            inst2.unLock();

            if (useMax)
                inst2.setLabeling(meModel.classify(inst).getLabelVector().getBestLabel());
            else
                inst2.setLabeling(meModel.classify(inst).getLabeling());

            inst2.lock();
            ret.add(inst2, 0.1);
        }
        return ret;
    }

    public void stop(int exitCode)
    {
        System.exit(exitCode);
    }
}
