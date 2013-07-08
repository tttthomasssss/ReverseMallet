package dualist.classify;

import cc.mallet.classify.*;
import cc.mallet.classify.constraints.ge.MaxEntFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntGEConstraint;
import cc.mallet.classify.constraints.ge.MaxEntKLFLGEConstraints;
import cc.mallet.classify.constraints.ge.MaxEntL2FLGEConstraints;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Multinomial;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

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
    Multinomial.Estimator priorEstimator = new Multinomial.LaplaceEstimator();

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
    private int numIterations = 0;
    private int maxIterations = Integer.MAX_VALUE;

    protected ArrayList<MaxEntGEConstraint> constraints;
    private String constraintsFile;

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

    public MaxEnt train(InstanceList trainingList)
    {
        return this.train(trainingList, maxIterations);
    }

    public MaxEnt train(InstanceList trainingList, int maxIterations)
    {
        if (constraints == null && constraintsFile != null) {

            HashMap<Integer, double[]> constraintsMap = FeatureConstraintUtil.readConstraintsFromFile(constraintsFile, trainingList);

            constraints = new ArrayList<MaxEntGEConstraint>();

            MaxEntFLGEConstraints geConstraints = (l2) ? new MaxEntL2FLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues, normalize) : new MaxEntKLFLGEConstraints(trainingList.getDataAlphabet().size(), trainingList.getTargetAlphabet().size(), useValues);

            for (int fi : constraintsMap.keySet()) {
                geConstraints.addConstraint(fi, constraintsMap.get(fi), 1);
            }
            constraints.add(geConstraints);
        }

        this.setupEstimationEngine(trainingList);

        // Estimate Label Priors
        // We are solely interested in the labeled Instances here, the labeled Features are incorporated in the model just fine, its the instances stupid!
        //check the incorporateOneInstance method of how to incorporate one more instance into the model and update the expectations, then use these expectations to derive constraint probabilites!!!
        Multinomial estimation = pe.estimate();
        //Multinomial[] featureEstimation =
        //this.estimateConstraints()

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
    /*
    private Multinomial[] estimateFeatureMultinomials () {
        int numLabels = targetAlphabet.size();
        Multinomial[] m = new Multinomial[numLabels];

        // first add all the appropriate pseudocounts (derived from feature labels)
        // to the multinomial estimator
        if (labelFeatures != null) {
            filterlabelFeatures();
            for (Object label : labelFeatures.keySet()) {

                int li = targetAlphabet.lookupIndex(label);
                for (Object feature : labelFeatures.get(label)) {

                    int fi = dataAlphabet.lookupIndex(feature);

                    System.out.println("MULTINOMIAL ESTIMATION OF FEATURE[" + feature + "] WITH LABEL[" + label + "] BEFORE INCREMENTING WITH ALPHA[" + alpha + "]: " + me[li].getCount(fi));
                    me[li].increment(fi, alpha);
                    System.out.println("MULTINOMIAL ESTIMATION OF FEATURE[" + feature + "] WITH LABEL[" + label + "] AFTER INCREMENTING WITH ALPHA[" + alpha + "]: " + me[li].getCount(fi));
                }
            }
        }

        System.out.println("########################################### START");
        // now estimate conditionals from data

        System.out.println("### NUMLABELS: " + numLabels);
        System.out.println("### M SIZE: " + m.length);

        for (int li = 0; li < numLabels; li++) {
            m[li] = me[li].estimate();
        }

        System.out.println("########################################### END");

        return m;
    }
      */

    //-- Getter/Setter Business...so much ado 'bout nuthin' --//

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
