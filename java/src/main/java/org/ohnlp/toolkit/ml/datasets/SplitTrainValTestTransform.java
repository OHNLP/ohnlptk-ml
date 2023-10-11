package org.ohnlp.toolkit.ml.datasets;

import org.apache.beam.sdk.schemas.Schema;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.values.*;
import org.ohnlp.backbone.api.components.OneToManyTransform;
import org.ohnlp.backbone.api.exceptions.ComponentInitializationException;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


public class SplitTrainValTestTransform extends OneToManyTransform {
    private double trainWeight;
    private double testWeight;
    private double validationWeight;
    @Override
    public Map<String, Schema> calculateOutputSchema(Schema schema) {
        return Map.of("Train", schema, "Validation", schema, "Test", schema);
    }

    @Override
    public PCollectionRowTuple expand(PCollection<Row> input) {
        Schema outSchema = input.getSchema();
        double trainThreshold = trainWeight;
        double testThreshold = trainWeight + testWeight;

        PCollectionTuple out = input.apply("Split into Train/Test/Validation", ParDo.of(
                new DoFn<Row, Row>() {
                    private SecureRandom random;

                    @Setup
                    public void init() {
                        this.random = new SecureRandom();
                    }

                    @ProcessElement
                    public void process(ProcessContext pc) {
                        double weight = this.random.nextDouble();
                        String tag = weight < trainThreshold ? "Train" : (weight < testThreshold ? "Test" : "Validation");
                        pc.output(new TupleTag<>(tag), weight);
                    }
                }
        ).withOutputTags(new TupleTag<>("Train"), TupleTagList.of(new TupleTag<>("Test")).and(new TupleTag<>("Validation"))));
        for (String tag : getOutputTags()) {
            out.get(tag).setRowSchema(outSchema);
        }
        return PCollectionRowTuple.of(
                "Train", out.get("Train"),
                "Validation", out.get("Validation"),
                "Test", out.get("Test")
        );
    }

    @Override
    public void init() throws ComponentInitializationException {
        if (trainWeight + validationWeight + testWeight != 1.0) {
            throw new ComponentInitializationException(new IllegalArgumentException("Sum of train, validation, and test weights must equal 1"));
        }
    }

    @Override
    public List<String> getOutputTags() {
        return Arrays.asList("Train", "Validation", "Test");
    }
}
