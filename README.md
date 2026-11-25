# Document Classifier MVP

A small Go project that classifies text extracted from documents using a Multinomial Naive Bayes model.

## Current scope

- Multinomial Naive Bayes trained from labeled text examples.
- Tokenization with basic lowercasing, punctuation removal, and stop-word filtering.
- Laplace/additive smoothing controlled by `alpha`.
- Prediction with a normalized model score.
- Review flags for empty input, explicit truncation markers, suspicious patterns, and low model scores.
- Unit tests for classification, invalid training data, and model state integrity.

## Document types

- `TEXTUAL`
- `LEGAL`
- `FINANCIAL`
- `GRAPHICAL`
- `UNKNOWN` (fallback when the model cannot predict)

## Run

Requires Go 1.25.6 or compatible.

```bash
go test ./...
go run ./cmd/demo
```

## Train the model

Configure the labels, then pass labeled examples to `Train`:

```go
classifier := document.NewNaiveBayes(
    1.0,
    document.DocTypeFinancial,
    document.DocTypeLegal,
    document.DocTypeTextual,
)

err := classifier.Train([]document.Example{
    {
        Label: document.DocTypeFinancial,
        Text:  "Factura total a pagar vencimiento IVA",
    },
    {
        Label: document.DocTypeLegal,
        Text:  "Contrato cláusula jurisdicción obligaciones",
    },
})
if err != nil {
    log.Fatal(err)
}
```

`Train` validates the entire batch before modifying the model. It returns an error for an empty batch, an unconfigured or unknown label, text with no usable tokens, or an invalid smoothing parameter.

## Review flags

- `Unreadable`: the input is empty or contains only whitespace.
- `Incomplete`: the input contains an explicit truncation marker, such as `[truncated]`, `[truncado]`, or `documento truncado`.
- `Suspicious`: the rule-based scanner finds one of its configured suspicious patterns.
- `NeedsReview`: at least one review condition is met, including low model score.

Text length alone does not establish document completeness. This MVP does not yet infer truncation from document structure, page metadata, or extraction diagnostics. Absence of an explicit marker does not guarantee that a document is complete.

## Important limitations

- The included demo uses a tiny illustrative dataset. It is not evidence of generalization.
- The returned score is normalized across the model's class scores; it is **not a calibrated probability**.
- The current classifier works on text already provided to it. PDF/image ingestion, asynchronous processing, persistence, and production deployment are outside this MVP's implemented scope.

## Next engineering steps

1. Build a larger, representative, labeled dataset.
2. Evaluate on held-out data using precision, recall, F1-score, and a confusion matrix.
3. Compare against a majority-class baseline.
4. Improve tokenization and document-specific preprocessing.
5. Calibrate prediction scores before treating them as probabilities.
