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

import java.io.Serializable;
import java.util.ArrayList;
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
    Multinomial.Estimator priorEstimator = new Multinomial.LaplaceEstimator();//new Multinomial.MLEstimator();

    Multinomial.Estimator[] me;
    Multinomial.Estimator pe;

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
    private double constraintWeight = 1.0;
    private double docLengthNormalization = -1;  // A value of -1 means don't do any document length normalization
    private int numIterations = 0;
    private int maxIterations = Integer.MAX_VALUE;

    protected ArrayList<MaxEntGEConstraint> constraints;
    private String constraintsFile;

    public MaxEntGEWithPriorsTrainer()
    {
        super();
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
        this.constraints = (constraints == null && constraintsFile != null) ? this.createConstraintsFromFile(trainingList) : this.createConstraints(trainingList, labeledFeatures, this.defaultMajorityProb);

        this.setupEstimationEngine(trainingList);

        if (labeledSet.size() > 0) {
            for (Instance labeledInstance : labeledSet) {
                this.incorporateOneInstance(labeledInstance, trainingList.getInstanceWeight(labeledInstance), labeledFeatures);
            }

            HashMultimap<Integer, String> labeledInstances = this.addConstraintEstimations(trainingList, labeledFeatures);

            this.constraints.addAll(this.createConstraints(trainingList, labeledInstances, 0.8));
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
            pe = (Multinomial.Estimator) priorEstimator.clone();
            pe.setAlphabet(trainingList.getTargetAlphabet());
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

            me[li].increment (fv, labelWeight * instanceWeight);
            // This relies on labelWeight summing to 1 over all labels
            pe.increment (li, labelWeight * instanceWeight);
        }
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
                        fv.setValue(idx, fv.value(idx) + 50);
                    }
                }
            }
        }
        return fv;
    }

    private HashMultimap<Integer, String> addConstraintEstimations(InstanceList trainingList, HashMultimap<Integer, String> labeledFeatures)
    {
        Multinomial[] m = new Multinomial[trainingList.getTargetAlphabet().size()];

        for (int labelIndex = 0; labelIndex < trainingList.getTargetAlphabet().size(); labelIndex++) {
            m[labelIndex] = me[labelIndex].estimate();
        }

        HashMultimap<Integer, String> labeledInstances = HashMultimap.create();

        // TODO: This is still very ugly because its restricted to binary decisions
        for (int i = 0; i < trainingList.getDataAlphabet().size(); i++) {
            if (Math.abs(m[0].value(i) - m[1].value(i)) > 0.001 && !labeledFeatures.get((m[0].value(i) > m[1].value(i) ? 0 : 1)).contains(trainingList.getDataAlphabet().lookupObject(i))) {
                labeledInstances.put((m[0].value(i) > m[1].value(i) ? 0 : 1), (String)trainingList.getDataAlphabet().lookupObject(i));
            }
        }

        return labeledInstances;
    }

    //-- Getter/Setter Business...so much ado 'bout nuthin' --//

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
