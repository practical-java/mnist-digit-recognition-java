package com.practicaldeveloper.mnist;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.eval.Evaluation;

public class DL4JMnistDemo {
    public static void main(String[] args) throws Exception {
        DataSetIterator mnistTrain = new MnistDataSetIterator(64, true, 123);
        DataSetIterator mnistTest = new MnistDataSetIterator(64, false, 123);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new org.nd4j.linalg.learning.config.Nesterovs(0.01, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(1000).nOut(10)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        System.out.println("Training model...");
        for (int i = 0; i < 5; i++) {
            model.fit(mnistTrain);
        }

        System.out.println("Evaluating model...");
        Evaluation eval = model.evaluate(mnistTest);
        System.out.println(eval.stats());
    }
}
