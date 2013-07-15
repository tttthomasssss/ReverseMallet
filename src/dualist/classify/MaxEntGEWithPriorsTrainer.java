package dualist.classify;

import cc.mallet.classify.*;
import cc.mallet.classify.constraints.ge.MaxEntFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntGEConstraint;
import cc.mallet.classify.constraints.ge.MaxEntKLFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntL2FLGEConstraints;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;
import cc.mallet.types.*;
import com.google.common.collect.HashMultimap;
import dualist.tui.Util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

/**
 * Created with IntelliJ IDEA.
 * User: thk22
 * Date: 05/07/2013
 * Time: 14:56
 * To change this template use File | Settings | File Templates.
 */
public class MaxEntGEWithPriorsTrainer
        extends ClassifierTrainer<MaxEnt>
        implements ClassifierTrainer.ByOptimization<MaxEnt>, Boostable, Serializable
{
    // These function as default selections for the kind of Estimator used
    // Other than Settles, I am trying to use the estimates to bootstrap constraints for the model
    // Further, instead of LaplaceEstimators I use MLEstimators which have a default value of 0 rather than 1
    Multinomial.Estimator featureEstimator = new Multinomial.MLEstimator();
    //Multinomial.Estimator priorEstimator = new Multinomial.LaplaceEstimator();//new Multinomial.MLEstimator();

    Multinomial.Estimator[] me;
    //Multinomial.Estimator pe;

    // All the parameterisation stuff
    private Optimizer optimizer = null;
    private MaxEntOptimizableByGE generalizedExpectation = null;
    private InstanceList trainingList;
    private MaxEnt classifier;

    private boolean l2 = false;
    private boolean normalize = true;
    private boolean useValues = false;

    private double temperature = 1.0;
    private double gaussianPriorVariance = 1.0;
    private double defaultMajorityProb = 0.9;
    private double instMajorityProb = 0.7;
    private double biasInstanceWeight = 1.5;
    private double biasInstanceNoFeatureWeight = 1.2;
    private double deltaThreshold = 0.002;
    private double constraintWeight = 1.0;
    private double docLengthNormalization = -1;  // A value of -1 means don't do any document length normalization
    private int numIterations = 0;
    private int maxIterations = Integer.MAX_VALUE;

    private double[][] globalFeatureCounts;
    private PerLabelInfoGain globalInfoGain;

    protected ArrayList<MaxEntGEConstraint> constraints;
    private String constraintsFile;

    public MaxEntGEWithPriorsTrainer()
    {
        this(null);
    }

    public MaxEntGEWithPriorsTrainer(ArrayList<MaxEntGEConstraint> constraints)
    {
        super();
        this.constraints = constraints;

    }

    public Optimizable.ByGradientValue getOptimizable (InstanceList trainingList) {
        if (generalizedExpectation == null) {
            generalizedExpectation = new MaxEntOptimizableByGE(trainingList, constraints, classifier);
            generalizedExpectation.setTemperature(temperature);
            generalizedExpectation.setGaussianPriorVariance(gaussianPriorVariance);
        }
        return generalizedExpectation;
    }

    public Optimizer getOptimizer()
    {
        getOptimizable(trainingList);

        if (optimizer == null) {
            optimizer = new LimitedMemoryBFGS(generalizedExpectation);
        }

        return optimizer;
    }

    public int getIteration()
    {
        return numIterations;
    }

    public MaxEnt train(InstanceList trainingList, InstanceList labeledSet, HashMultimap<Integer, String> labeledFeatures)
    {
        return this.train(trainingList, labeledSet, labeledFeatures, maxIterations);
    }

    public MaxEnt train(InstanceList trainingList, InstanceList labeledSet, HashMultimap<Integer, String> labeledFeatures, int maxIterations)
    {
        this.setupEstimationEngine(trainingList);

        if (labeledSet.size() > 0) {
            //for (Instance labeledInstance : labeledSet) {
            //    this.incorporateOneInstance(labeledInstance, trainingList.getInstanceWeight(labeledInstance), labeledFeatures);
            //}

            // Try some stuff with infogain
            // Calc Labels first
            // Hide Labels for the MaxEnts

            //this.globalFeatureCounts = this.calcFeatureCounts(trainingList);

            // DIRTY HACKHACKHACKHACKHACKHACKHACKHACK
            trainingList = Util.hideAllLabels(trainingList);

            double[][] labeledSetFeatCount = this.calcFeatureCounts(labeledSet);
            PerLabelInfoGain labeledSetInfoGain = new PerLabelInfoGain(labeledSet);

            for (int i = 0; i < trainingList.getTargetAlphabet().size(); i++) {
                for (int j = 0; j < Math.min(labeledSet.size() * 4, 100); j++) {
                    String feature = (String)labeledSetInfoGain.getInfoGain(i).getObjectAtRank(j);
                    int index = labeledSetInfoGain.getInfoGain(i).getIndexAtRank(j);

                    //double maxVal = Math.max(this.globalFeatureCounts[0][index], this.globalFeatureCounts[1][index]);
                    //double delta = Math.abs(this.globalFeatureCounts[0][index] - this.globalFeatureCounts[1][index]);

                    double maxVal = Math.max(labeledSetFeatCount[0][index], labeledSetFeatCount[1][index]);
                    double delta = labeledSetFeatCount[i][index] - labeledSetFeatCount[1 - i][index];

                    //if (labeledSetFeatCount[i][index] > maxVal * 0.75 && delta > maxVal * 0.75) {
                    if (delta > maxVal * 0.75) {
                        labeledFeatures.put(i, feature);
                    }
                    System.out.println("FEATURE=[" + feature + "] FOR LABEL[" + trainingList.getTargetAlphabet().lookupObject(i) + "]; CURR CORRLEATION VALUE=[" + labeledSetFeatCount[i][index] + "]; GLOBAL MAXIMUM=[" + maxVal + "]");
                }
            }

            //HashMultimap<Integer, String> labeledInstances = this.estimateInstanceConstraints(trainingList, labeledFeatures);

            //this.constraints = (labeledFeatures == null && constraintsFile != null) ? this.createConstraintsFromFile(trainingList) : this.createAllConstraints(trainingList, labeledFeatures, labeledInstances, this.defaultMajorityProb, this.instMajorityProb);
            this.constraints = (labeledFeatures == null && constraintsFile != null) ? this.createConstraintsFromFile(trainingList) : this.createConstraints(trainingList, labeledFeatures, this.defaultMajorityProb);
        } else {
            this.constraints = (constraints == null && constraintsFile != null) ? this.createConstraintsFromFile(trainingList) : this.createConstraints(trainingList, labeledFeatures, this.defaultMajorityProb);
        }

        // Don't blame me for the code down there, been pretty much taken from the original MaxEntGERangeTrainer
        getOptimizable(trainingList);
        getOptimizer();

        if (optimizer instanceof LimitedMemoryBFGS) {
            ((LimitedMemoryBFGS)optimizer).reset();
        }

        try {
            optimizer.optimize(maxIterations);
            numIterations += maxIterations;
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (maxIterations == Integer.MAX_VALUE && optimizer instanceof LimitedMemoryBFGS) {
            // Run it again because in our and Sam Roweis' experience, BFGS can still
            // eke out more likelihood after first convergence by re-running without
            // being restricted by its gradient history.
            ((LimitedMemoryBFGS)optimizer).reset();
            try {
                optimizer.optimize(maxIterations);
                numIterations += maxIterations;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        classifier = generalizedExpectation.getClassifier();

        return classifier;
    }

    public MaxEnt train(InstanceList trainingList)
    {
        return this.train(trainingList, null, null, maxIterations);
    }

    public MaxEnt train(InstanceList trainingList, int maxIterations)
    {
        return this.train(trainingList, null, null, maxIterations);
    }

    public MaxEnt getClassifier()
    {
        return classifier;
    }

    private void setupEstimationEngine(InstanceList trainingList)
    {
        if (me == null) {
            int numLabels = trainingList.getTargetAlphabet().size();
            me = new Multinomial.Estimator[numLabels];
            for (int i = 0; i < numLabels; i++) {
                me[i] = (Multinomial.Estimator) featureEstimator.clone();
                me[i].setAlphabet(trainingList.getDataAlphabet());
            }
            //pe = (Multinomial.Estimator) priorEstimator.clone();
            //pe.setAlphabet(trainingList.getTargetAlphabet());
        }

        if (trainingList.getTargetAlphabet().size() > me.length) {
            // target alphabet grew. increase size of our multinomial array
            int targetAlphabetSize = trainingList.getTargetAlphabet().size();
            // copy over old values
            Multinomial.Estimator[] newMe = new Multinomial.Estimator[targetAlphabetSize];
            System.arraycopy (me, 0, newMe, 0, me.length);
            // initialize new expanded space
            for (int i= me.length; i<targetAlphabetSize; i++){
                Multinomial.Estimator mest = (Multinomial.Estimator)featureEstimator.clone ();
                mest.setAlphabet (trainingList.getTargetAlphabet());
                newMe[i] = mest;
            }
            me = newMe;
        }
    }

    private void incorporateOneInstance(Instance instance, double instanceWeight, HashMultimap<Integer, String> labeledFeatures)
    {
        Labeling labeling = instance.getLabeling ();
        if (labeling == null) return; // Handle unlabeled instances by skipping them
        FeatureVector fv = (FeatureVector) instance.getData ();

        double oneNorm = fv.oneNorm();
        if (oneNorm <= 0) return; // Skip instances that have no features present
        if (docLengthNormalization > 0)
            // Make the document have counts that sum to docLengthNormalization
            // I.e., if 20, it would be as if the document had 20 words.
            instanceWeight *= docLengthNormalization / oneNorm;
        assert (instanceWeight > 0 && !Double.isInfinite(instanceWeight));
        for (int lpos = 0; lpos < labeling.numLocations(); lpos++) {
            int li = labeling.indexAtLocation (lpos);
            double labelWeight = labeling.valueAtLocation (lpos);
            if (labelWeight == 0) continue;

            // Add a bias to every feature in a labeled instance in case they co occur with a labeled feature
            //fv = this.biasFeatureVector(fv, li, labeledFeatures);
            instanceWeight = this.biasInstanceWeight(fv, li, labeledFeatures);

            me[li].increment (fv, labelWeight * instanceWeight);
            // This relies on labelWeight summing to 1 over all labels
            //pe.increment (li, labelWeight * instanceWeight);
        }
    }

    private ArrayList<MaxEntGEConstraint> createAllConstraints(InstanceList trainingList, HashMultimap<Integer, String> labeledFeatures, HashMultimap<Integer, String> labeledInstances, double featMajorityProb, double instMajorityProb)
    {
        ArrayList<MaxEntGEConstraint> currConstraints = new ArrayList<MaxEntGEConstraint>();

        // L2 Penalty?
        MaxEntFLGEConstraints geConstraints = (l2) ? new MaxEntL2FLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues, normalize) : new MaxEntKLFLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues);

        // Clear & Rebuild
        double[] in_probs = {featMajorityProb, instMajorityProb};
        HashMultimap[] in_data = {labeledFeatures, labeledInstances};

        for (int i = 0; i < in_probs.length; i++) {

            double[] probs;
            double minorityProb = (1 - in_probs[i]) / (trainingList.getTargetAlphabet().size() - 1);

            for (Object key : in_data[i].keySet()) {

                int li = (Integer)key;

                if (li >= 0) {
                    // Fill the Constraints
                    probs = new double[trainingList.getTargetAlphabet().size()];

                    // Majority Probability for current Label
                    probs[li] = in_probs[i];

                    // Minority Probabilities for all the other Labels
                    for (int j = 0; j < li; j++) {
                        probs[j] = minorityProb;
                    }
                    for (int j = (li + 1); j < probs.length; j++) {
                        probs[j] =  minorityProb;
                    }

                    // Finally collect the constraints
                    for (Object featureObj : in_data[i].get(li)) {
                        String featureName = (String)featureObj;
                        geConstraints.addConstraint(trainingList.getDataAlphabet().lookupIndex(featureName, false), probs, constraintWeight);
                    }
                }
            }
        }

        currConstraints.add(geConstraints);

        return currConstraints;
    }

    private ArrayList<MaxEntGEConstraint> createConstraints(InstanceList trainingList, HashMultimap<Integer, String> labeledFeatures, double majorityProb)
    {
        ArrayList<MaxEntGEConstraint> currConstraints = new ArrayList<MaxEntGEConstraint>();

        // L2 Penalty?
        MaxEntFLGEConstraints geConstraints = (l2) ? new MaxEntL2FLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues, normalize) : new MaxEntKLFLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues);

        // Clear & Rebuild
        double[] probs;
        double minorityProb = (1 - majorityProb) / (trainingList.getTargetAlphabet().size() - 1);

        for (int li : labeledFeatures.keySet()) {

            if (li >= 0) {
                // Fill the Constraints
                probs = new double[trainingList.getTargetAlphabet().size()];

                // Majority Probability for current Label
                probs[li] = majorityProb;

                // Minority Probabilities for all the other Labels
                for (int i = 0; i < li; i++) {
                    probs[i] = minorityProb;
                }
                for (int i = (li + 1); i < probs.length; i++) {
                    probs[i] =  minorityProb;
                }

                // Finally collect the constraints
                for (String featureName : labeledFeatures.get(li)) {
                    geConstraints.addConstraint(trainingList.getDataAlphabet().lookupIndex(featureName, false), probs, constraintWeight);
                }
            }
        }

        currConstraints.add(geConstraints);

        return currConstraints;
    }

    private ArrayList<MaxEntGEConstraint> createConstraintsFromFile(InstanceList trainingList)
    {
        HashMap<Integer, double[]> constraintsMap = FeatureConstraintUtil.readConstraintsFromFile(constraintsFile, trainingList);

        ArrayList<MaxEntGEConstraint> currConstraints = new ArrayList<MaxEntGEConstraint>();

        MaxEntFLGEConstraints geConstraints = (l2) ? new MaxEntL2FLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues, normalize) : new MaxEntKLFLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues);

        for (int fi : constraintsMap.keySet()) {
            geConstraints.addConstraint(fi, constraintsMap.get(fi), 1);
        }

        currConstraints.add(geConstraints);

        return currConstraints;
    }

    private double[][] calcFeatureCounts(InstanceList instanceList)
    {
        double[] labelCounts = new double[instanceList.getTargetAlphabet().size()];
        double[][] featureCounts = new double[instanceList.getTargetAlphabet().size()][instanceList.getDataAlphabet().size()];

        for (Instance instance : instanceList) {
            FeatureVector fv = (FeatureVector)instance.getData();
            Labeling l = (Labeling)instance.getTarget();
            l.addTo(labelCounts);
            for (int li = 0; li < instance.getTargetAlphabet().size(); li++) {
                double val = l.value(li);
                fv.addTo(featureCounts[li], val);
            }
        }
        for (int li = 0; li < labelCounts.length; li++) {
            for (int fi = 0; fi < featureCounts[li].length; fi++) {
                featureCounts[li][fi] /= labelCounts[li];
            }
        }

        return featureCounts;
    }

    /**
     * If a feature in a labeled Instance co-occurs with a labeled Feature, we bias all the features in that Instance for every co-occurrence by a well-contemplated(=http://xkcd.com/221/) value
     * @param fv
     * @param labelIndex
     * @param labeledFeatures
     * @return
     */
    private FeatureVector biasFeatureVector(FeatureVector fv, int labelIndex, HashMultimap<Integer, String> labeledFeatures)
    {
        if (labelIndex >= 0) {
            for (String feature : labeledFeatures.get(labelIndex)) {
                if (fv.contains(feature)) {
                    for (int idx : fv.getIndices()) {
                        if (idx > 0) {
                            fv.setValue(idx, fv.value(idx) + 10);
                        }
                    }
                }
            }
        }
        return fv;
    }

    private double biasInstanceWeight(FeatureVector fv, int labelIndex, HashMultimap<Integer, String> labeledFeatures)
    {
        double weight = this.biasInstanceNoFeatureWeight;

        if (labelIndex >= 0) {
            for (String feature : labeledFeatures.get(labelIndex)) {
                if (fv.contains(feature)) {
                    weight = this.biasInstanceWeight;
                    break;
                }
            }
        }
        return weight;
    }

    private HashMultimap<Integer, String> estimateInstanceConstraints(InstanceList trainingList, HashMultimap<Integer, String> labeledFeatures)
    {
        Multinomial[] m = new Multinomial[trainingList.getTargetAlphabet().size()];

        for (int labelIndex = 0; labelIndex < trainingList.getTargetAlphabet().size(); labelIndex++) {
            m[labelIndex] = me[labelIndex].estimate();
        }

        HashMultimap<Integer, String> labeledInstances = HashMultimap.create();

        //ArrayList<Double> diff = new ArrayList<Double>();
        //double[] diff = new double[trainingList.getDataAlphabet().size()];
        //double[] diff01 = new double[trainingList.getDataAlphabet().size()];
        //double[] diff015 = new double[trainingList.getDataAlphabet().size()];
        //double[] diff02 = new double[trainingList.getDataAlphabet().size()];
        //double d = 0;

        //Do gehts weiter und wemma fertig san dann klatschma den schas ins TweetClassification Projekt und schaun wos passiert!
        // TODO: This is still very ugly because its restricted to binary decisions
        for (int i = 0; i < trainingList.getDataAlphabet().size(); i++) {
          //  d = Math.abs(m[0].value(i) - m[1].value(i));
          //  diff[i] = d;
          //  diff01[i] = d > 0.001 ? d : 0;
          //  diff015[i] = d > 0.0015 ? d : 0;
          //  diff02[i] = d > 0.002 ? d : 0;
            if (Math.abs(m[0].value(i) - m[1].value(i)) > this.deltaThreshold/*0.002*//*0.0015*/ && !labeledFeatures.get((m[0].value(i) > m[1].value(i) ? 0 : 1)).contains(trainingList.getDataAlphabet().lookupObject(i))) {
                labeledInstances.put((m[0].value(i) > m[1].value(i) ? 0 : 1), (String)trainingList.getDataAlphabet().lookupObject(i));
            }
        }

        /*
        Arrays.sort(diff);

        System.out.println("SMALLEST: " + diff[0]);
        System.out.println("LARGEST: " + diff[diff.length - 1]);
        System.out.println("DIFF 1/n: " + Math.abs(diff[0] - diff[diff.length - 1]));

        int mid1 = (int)Math.floor(diff.length / 2);
        int mid2 = (int)Math.floor((diff.length / 2) + 1);

        System.out.println("DIFF MID: " + Math.abs(diff[mid1] - diff[mid2]));

        int cnt = 0;
        int cnt01 = 0;
        int cnt015 = 0;
        int cnt02 = 0;
        for (int i = 0; i < diff.length; i++) {
            cnt += diff[i] == 0 ? 1 : 0;
            cnt01 += diff01[i] == 0 ? 1 : 0;
            cnt015 += diff015[i] == 0 ? 1 : 0;
            cnt02 += diff02[i] == 0 ? 1 : 0;
        }

        System.out.println("> 0: " + cnt + " OUT OF " + diff.length + " ARE 0! THATS " + ((double)cnt / (double)diff.length) + "%");
        System.out.println("> 0.01: " + cnt01 + " OUT OF " + diff01.length + " ARE 0! THATS " + ((double)cnt01 / (double)diff01.length) + "%");
        System.out.println("> 0.015: " + cnt015 + " OUT OF " + diff015.length + " ARE 0! THATS " + ((double)cnt015 / (double)diff015.length) + "%");
        System.out.println("> 0.02: " + cnt02 + " OUT OF " + diff02.length + " ARE 0! THATS " + ((double)cnt02 / (double)diff02.length) + "%");
          */

        return labeledInstances;
    }

    //-- Getter/Setter Business...so much ado 'bout nuthin' --//

    public double getBiasInstanceNoFeatureWeight()
    {
        return this.biasInstanceNoFeatureWeight;
    }

    public void setBiasInstanceNoFeatureWeight(double biasInstanceNoFeatureWeight)
    {
        this.biasInstanceNoFeatureWeight = biasInstanceNoFeatureWeight;
    }

    public double getInstMajorityProb()
    {
        return this.instMajorityProb;
    }

    public void setInstMajorityProb(double instMajorityProb)
    {
        this.instMajorityProb = instMajorityProb;
    }

    public double getBiasInstanceWeight()
    {
        return this.biasInstanceWeight;
    }

    public void setBiasInstanceWeight(double biasInstanceWeight)
    {
        this.biasInstanceWeight = biasInstanceWeight;
    }

    public double getDeltaThreshold()
    {
        return this.deltaThreshold;
    }

    public void setDeltaThreshold(double deltaThreshold)
    {
        this.deltaThreshold = deltaThreshold;
    }

    public double getConstraintWeight()
    {
        return this.constraintWeight;
    }

    public void setConstraintWeight(double constraintWeight)
    {
        this.constraintWeight = constraintWeight;
    }

    public double getMajorityProb()
    {
        return this.defaultMajorityProb;
    }

    public void setMajorityProb(double majorityProb)
    {
        this.defaultMajorityProb = majorityProb;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public String getConstraintsFile() {
        return constraintsFile;
    }

    public void setConstraintsFile(String constraintsFile) {
        this.constraintsFile = constraintsFile;
    }

    public double getGaussianPriorVariance() {
        return gaussianPriorVariance;
    }

    public void setGaussianPriorVariance(double gaussianPriorVariance) {
        this.gaussianPriorVariance = gaussianPriorVariance;
    }

    public double getTemperature() {
        return temperature;
    }

    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    public boolean isUseValues() {
        return useValues;
    }

    public void setUseValues(boolean useValues) {
        this.useValues = useValues;
    }

    public boolean isNormalize() {
        return normalize;
    }

    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
    }

    public boolean isL2() {
        return l2;
    }

    public void setL2(boolean l2) {
        this.l2 = l2;
    }
}
