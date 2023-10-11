package org.ohnlp.toolkit.ml.feat;

import org.apache.beam.sdk.coders.BigEndianLongCoder;
import org.apache.beam.sdk.coders.DoubleCoder;
import org.apache.beam.sdk.coders.KvCoder;
import org.apache.beam.sdk.coders.RowCoder;
import org.apache.beam.sdk.schemas.Schema;
import org.apache.beam.sdk.schemas.transforms.Join;
import org.apache.beam.sdk.schemas.transforms.Select;
import org.apache.beam.sdk.transforms.*;
import org.apache.beam.sdk.values.*;
import org.ohnlp.backbone.api.annotations.ComponentDescription;
import org.ohnlp.backbone.api.annotations.ConfigurationProperty;
import org.ohnlp.backbone.api.annotations.InputColumnProperty;
import org.ohnlp.backbone.api.components.OneToManyTransform;
import org.ohnlp.backbone.api.config.InputColumn;
import org.ohnlp.backbone.api.exceptions.ComponentInitializationException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@ComponentDescription(
        name = "Calculate TF-IDF",
        desc = "Calculates TF-IDFs for individual terms, both globally and stratified by a user-supplied list of columns"
)
public class TFIDFTransform extends OneToManyTransform {
    @ConfigurationProperty(
            path = "term",
            desc = "The column to use as a \"term\""
    )
    private InputColumn termColumn;
    @ConfigurationProperty(
            path = "doc",
            desc = "The column to use as a \"document\""
    )
    private InputColumn documentColumn;

    @ConfigurationProperty(
            path = "stratification",
            desc = "The columns to use for stratification. Defaults to empty",
            required = false
    )
    @InputColumnProperty
    private List<InputColumn> stratificationColumns = new ArrayList<>();

    @Override
    public PCollectionRowTuple expand(PCollection<Row> input) {
        // Calculate N
        PCollectionView<Long> collSize = input.apply("Calculate Collection Size: Select 'Doc IDs'", Select.fieldNames(documentColumn.getSourceColumnName()))
                .apply("Calculate Corpus Size: Distinct 'Doc IDs'", Distinct.create())
                .apply("Calculate Corpus Size: Count 'Doc IDs'", Count.globally()).apply(View.asSingleton());
        // Calculate |{d in D : t in D}| -- number of documents containing t
        Schema termSchema = Schema.of(
                input.getSchema().getField(termColumn.getSourceColumnName())
        );
        Schema docSchema = Schema.of(
                input.getSchema().getField(documentColumn.getSourceColumnName())
        );
        PCollection<KV<Row, Long>> documentsWithTerm = input.apply("Calculate Number of Documents per Term: Map to (term, docID)", MapElements.<Row, KV<Row, Row>>via(
                new SimpleFunction<>() {
                    @Override
                    public KV<Row, Row> apply(Row input) {
                        Row keyRow = Row.withSchema(termSchema).addValue(input.getValue(termColumn.getSourceColumnName())).build();
                        Row valueRow = Row.withSchema(docSchema).addValue(input.getValue(documentColumn.getSourceColumnName())).build();
                        return KV.of(keyRow, valueRow);
                    }
                })
        ).setCoder(KvCoder.of(RowCoder.of(termSchema), RowCoder.of(docSchema))
        ).apply("Calculate Number of Documents per Term: Distinct (term, docID)", Distinct.create()
        ).setCoder(KvCoder.of(RowCoder.of(termSchema), RowCoder.of(docSchema))
        ).apply("Calculate Number of Documents per Term: Count distinct docIDs per term", Count.perKey()
        ).setCoder(KvCoder.of(RowCoder.of(termSchema), BigEndianLongCoder.of()));
        // Calculate IDF by document and remap back to rows
        Schema idfSchema = Schema.of(
                input.getSchema().getField(termColumn.getSourceColumnName()),
                Schema.Field.of("idf", Schema.FieldType.DOUBLE)
        );
        PCollection<Row> idfByTerm = documentsWithTerm.apply("Calculate IDF Per Term", ParDo.of(new DoFn<KV<Row, Long>, KV<Row, Double>>() {
            @ProcessElement
            public void process(ProcessContext pc) {
                KV<Row, Long> e = pc.element();
                Long collSizeVal = pc.sideInput(collSize);
                double idf = Math.log((collSizeVal + 1.) / (e.getValue().doubleValue() + 1.)); // We use the modification with +1 to both numerator and denominator to avoid div-by-zero for terms that never occur (shouldn't ever be the case, but just for safety)
                pc.output(KV.of(e.getKey(), idf));
            }
        }).withSideInputs(collSize)
        ).setCoder(KvCoder.of(RowCoder.of(termSchema), DoubleCoder.of())
        ).apply("Calculate IDF per Term: Map back to Row for Joining", MapElements.via(new SimpleFunction<KV<Row, Double>, Row>() {
            @Override
            public Row apply(KV<Row, Double> input) {
                return Row.withSchema(idfSchema).addValues(input.getKey().getValues()).addValue(input.getValue()).build();
            }
        })
        ).setRowSchema(idfSchema);
        // Calculate TF both globally and by stratification columns. We do this by mapping each key to 1, and then summing by key
        List<Schema.Field> fields = new ArrayList<>();
        fields.add(input.getSchema().getField(termColumn.getSourceColumnName()));
        fields.add(input.getSchema().getField(documentColumn.getSourceColumnName()));
        Schema unstratifiedKeySchema = Schema.of(fields.toArray(Schema.Field[]::new));
        for (InputColumn stratificationColumn : stratificationColumns) {
            fields.add(input.getSchema().getField(stratificationColumn.getSourceColumnName()));
        }
        Schema stratifiedKeySchema = Schema.of(fields.toArray(Schema.Field[]::new));
        PCollectionTuple termCounts
                = input.apply("Calculate TF: Map to (Term + Doc ID + Stratification Columns, 1)", ParDo.of(
                new DoFn<Row, KV<Row, Long>>() {
                    @ProcessElement
                    public void process(ProcessContext pc) {
                        Row input = pc.element();
                        List<Object> values = stratifiedKeySchema.getFields().stream().map(f -> input.getValue(f.getName())).collect(Collectors.toList());
                        Row keyRow = Row.withSchema(stratifiedKeySchema).addValues(values).build();
                        pc.output(new TupleTag<>("stratified"), KV.of(keyRow, 1L));
                        values = unstratifiedKeySchema.getFields().stream().map(f -> input.getValue(f.getName())).collect(Collectors.toList());
                        keyRow = Row.withSchema(unstratifiedKeySchema).addValues(values).build();
                        pc.output(new TupleTag<>("unstratified"), KV.of(keyRow, 1L));
                    }
                }
        ).withOutputTags(new TupleTag<>("stratified"), TupleTagList.of(new TupleTag<>("unstratified"))));
        PCollection<KV<Row, Long>> stratifiedTF = termCounts.<KV<Row, Long>>get(
                "stratified"
        ).setCoder(KvCoder.of(RowCoder.of(stratifiedKeySchema), BigEndianLongCoder.of())
        ).apply("Calculate Stratified TF: Sum by Key", Sum.longsPerKey()
        ).setCoder(KvCoder.of(RowCoder.of(stratifiedKeySchema), BigEndianLongCoder.of()));
        PCollection<KV<Row, Long>> unstratifiedTF = termCounts.<KV<Row, Long>>get(
                "unstratified"
        ).setCoder(KvCoder.of(RowCoder.of(unstratifiedKeySchema), BigEndianLongCoder.of())
        ).apply("Calculate Unstratified TF: Sum by Key", Sum.longsPerKey()
        ).setCoder(KvCoder.of(RowCoder.of(unstratifiedKeySchema), BigEndianLongCoder.of()));
        // Join TFs with idfByTerm to get final TF-IDFs
        // - Map TFs back to rows in prep for joining
        ArrayList<Schema.Field> stratifiedFields = new ArrayList<>(stratifiedKeySchema.getFields());
        stratifiedFields.add(Schema.Field.of("tf", Schema.FieldType.INT64));
        ArrayList<Schema.Field> unstratifiedFields = new ArrayList<>(unstratifiedKeySchema.getFields());
        unstratifiedFields.add(Schema.Field.of("tf", Schema.FieldType.INT64));
        Schema stratifiedTFSchema = Schema.of(stratifiedFields.toArray(Schema.Field[]::new));
        Schema unstratifiedTFSchema = Schema.of(unstratifiedFields.toArray(Schema.Field[]::new));
        PCollection<Row> stratifiedTFRows = stratifiedTF.apply("Calculate Stratified TF-IDF: Map KV Back to Row", MapElements.via(new SimpleFunction<KV<Row, Long>, Row>() {
            @Override
            public Row apply(KV<Row, Long> input) {
                return Row.withSchema(stratifiedTFSchema).addValues(input.getKey().getValues()).addValue(input.getValue()).build();
            }
        })).setCoder(RowCoder.of(stratifiedTFSchema));
        PCollection<Row> unstratifiedTFRows = unstratifiedTF.apply("Calculate Unstratified TF-IDF: Map KV Back to Row", MapElements.via(new SimpleFunction<KV<Row, Long>, Row>() {
            @Override
            public Row apply(KV<Row, Long> input) {
                return Row.withSchema(unstratifiedTFSchema).addValues(input.getKey().getValues()).addValue(input.getValue()).build();
            }
        })).setCoder(RowCoder.of(unstratifiedTFSchema));
        // - And then do the actual join
        stratifiedFields.add(Schema.Field.of("idf", Schema.FieldType.DOUBLE));
        stratifiedFields.add(Schema.Field.of("tfidf", Schema.FieldType.DOUBLE));
        unstratifiedFields.add(Schema.Field.of("idf", Schema.FieldType.DOUBLE));
        unstratifiedFields.add(Schema.Field.of("tfidf", Schema.FieldType.DOUBLE));
        Schema stratifiedTFIDFSchema = Schema.of(stratifiedFields.toArray(Schema.Field[]::new));
        Schema unstratifiedTFIDFSchema = Schema.of(stratifiedFields.toArray(Schema.Field[]::new));
        PCollection<Row> stratifiedTFIDF = stratifiedTFRows.apply(
                "Calculate Stratified TF-IDF: Join TF and IDF Rows",
                Join.<Row, Row>innerJoin(idfByTerm).using(termColumn.getSourceColumnName())
        ).apply(
                "Calculate Stratified TF-IDF: Multiply TF and IDF and Flatten Joined Rows",
                MapElements.via(new SimpleFunction<Row, Row>() {
                    @Override
                    public Row apply(Row input) {
                        Row tfRow = input.getRow("lhs");
                        Row idfRow = input.getRow("lhs");
                        double tfIDF = tfRow.getInt64("tf").doubleValue() * idfRow.getDouble("idf");
                        return Row.withSchema(stratifiedTFIDFSchema).addValues(tfRow.getValues()).addValue(idfRow.getValue("idf")).addValue(tfIDF).build();
                    }
                })
        ).setRowSchema(stratifiedTFIDFSchema);
        PCollection<Row> unstratifiedTFIDF = unstratifiedTFRows.apply(
                "Calculate Unstratified TF-IDF: Join TF and IDF Rows",
                Join.<Row, Row>innerJoin(idfByTerm).using(termColumn.getSourceColumnName())
        ).apply(
                "Calculate Unstratified TF-IDF: Multiply TF and IDF and Flatten Joined Rows",
                MapElements.via(new SimpleFunction<Row, Row>() {
                    @Override
                    public Row apply(Row input) {
                        Row tfRow = input.getRow("lhs");
                        Row idfRow = input.getRow("lhs");
                        double tfIDF = tfRow.getInt64("tf").doubleValue() * idfRow.getDouble("idf");
                        return Row.withSchema(unstratifiedTFIDFSchema).addValues(tfRow.getValues()).addValue(idfRow.getValue("idf")).addValue(tfIDF).build();
                    }
                })
        ).setRowSchema(unstratifiedTFIDFSchema);
        return PCollectionRowTuple.of("Stratified TF-IDFs", stratifiedTFIDF).and("Global TF-IDFs", unstratifiedTFIDF);
    }

    @Override
    public void init() throws ComponentInitializationException {

    }

    @Override
    public List<String> getOutputTags() {
        return Arrays.asList("Stratified TF-IDFs", "Global TF-IDFs");
    }

    @Override
    public Map<String, Schema> calculateOutputSchema(Schema schema) {
        List<Schema.Field> stratifiedFields = new ArrayList<>();
        stratifiedFields.add(schema.getField(termColumn.getSourceColumnName()));
        stratifiedFields.add(schema.getField(documentColumn.getSourceColumnName()));
        for (InputColumn stratificationColumn : stratificationColumns) {
            stratifiedFields.add(schema.getField(stratificationColumn.getSourceColumnName()));
        }
        stratifiedFields.add(Schema.Field.of("tf", Schema.FieldType.INT64));
        stratifiedFields.add(Schema.Field.of("idf", Schema.FieldType.DOUBLE));
        stratifiedFields.add(Schema.Field.of("tfidf", Schema.FieldType.DOUBLE));
        Schema stratifiedSchema = Schema.of(stratifiedFields.toArray(Schema.Field[]::new));
        List<Schema.Field> unstratifiedFields = new ArrayList<>();
        unstratifiedFields.add(schema.getField(termColumn.getSourceColumnName()));
        unstratifiedFields.add(schema.getField(documentColumn.getSourceColumnName()));
        unstratifiedFields.add(Schema.Field.of("tf", Schema.FieldType.INT64));
        unstratifiedFields.add(Schema.Field.of("idf", Schema.FieldType.DOUBLE));
        unstratifiedFields.add(Schema.Field.of("tfidf", Schema.FieldType.DOUBLE));
        Schema globalSchema = Schema.of(unstratifiedFields.toArray(Schema.Field[]::new));
        return Map.of("Stratified TF-IDFs", stratifiedSchema, "Global TF-IDFs", globalSchema);
    }
}
