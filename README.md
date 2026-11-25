# Document Classifier MVP

A Go service that classifies text extracted from documents using a Multinomial Naive Bayes model.

## Current scope

- Multinomial Naive Bayes trained from labeled text examples.
- Tokenization with lowercasing, punctuation normalization, and stop-word filtering.
- Laplace/additive smoothing controlled by `alpha`.
- Per-label normalized model scores (not calibrated probabilities).
- Review flags for empty input, explicit truncation markers, suspicious patterns, and low model scores.
- JSONL dataset validation and family-aware offline evaluation.
- REST API for health checks and synchronous text classification.

## Document types

- `TEXTUAL`
- `LEGAL`
- `FINANCIAL`
- `GRAPHICAL`
- `UNKNOWN` (returned when the model cannot make a usable prediction)

## Requirements

Go 1.25.6 or compatible.

## Run tests and demo

```bash
go test ./...
go vet ./...
go run ./cmd/demo
```

## Run the REST API

The API trains the model once at startup from a validated JSONL dataset. By default it uses `data/documents.jsonl`.

PowerShell:

```powershell
$env:HTTP_ADDR = ":8080"
$env:DATASET_PATH = "data/documents.jsonl"
go run ./cmd/api
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `HTTP_ADDR` | `:8080` | HTTP listen address |
| `DATASET_PATH` | `data/documents.jsonl` | Labeled JSONL dataset used to train the model |

The process exits at startup if the dataset cannot be opened, parsed, validated, or used to train the model. The server uses read/write/idle timeouts and graceful shutdown.

### Health check

```http
GET /healthz
```

Example response:

```json
{"status":"ok"}
```

### Classify text

```http
POST /v1/classify
Content-Type: application/json

{"text":"Factura total a pagar $32000. Vencimiento e IVA."}
```

Example response shape:

```json
{
  "label": "FINANCIAL",
  "score": 0.82,
  "scores": {
    "FINANCIAL": 0.82,
    "GRAPHICAL": 0.04,
    "LEGAL": 0.08,
    "TEXTUAL": 0.06
  },
  "flags": {
    "unreadable": false,
    "incomplete": false,
    "suspicious": false,
    "needs_review": false
  }
}
```

The numeric values above are illustrative. Actual values depend on the selected training dataset.

The endpoint accepts one JSON object with a `text` string and rejects unknown fields, malformed or trailing JSON, unsupported content types, and request bodies larger than 1 MiB. It does not log submitted document text.

### HTTP status codes

- `200`: classification completed.
- `400`: malformed request JSON or invalid request shape.
- `404`: unknown route.
- `405`: unsupported HTTP method.
- `413`: request body exceeds 1 MiB.
- `415`: content type is not `application/json`.
- `503`: classifier is not initialized.

## Train the model in Go

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

## Evaluation

Run the family-aware evaluation command with a JSONL dataset:

```bash
go run ./cmd/evaluate data/documents.jsonl
```

The evaluator keeps documents from the same family in one split, then reports accuracy, macro precision/recall/F1, a confusion matrix, and a majority-class baseline.

## Important limitations

- The default dataset contains only a small number of illustrative records. It is not evidence of generalization.
- A larger synthetic validation dataset was evaluated during development, but synthetic test performance does not establish performance on real documents.
- Returned scores are normalized across class scores; they are **not calibrated probabilities**.
- The API classifies text already provided to it. It does not ingest PDFs or images, perform OCR, persist requests, authenticate clients, apply rate limits, or provide production deployment infrastructure.
- The `GRAPHICAL` label describes text about graphs/diagrams; the model does not analyze image pixels.

## Next engineering steps

1. Validate on representative, permissioned, anonymized real documents.
2. Add API authentication, rate limiting, structured operational metrics, and deployment configuration before exposing the service publicly.
3. Add persistence and traceability with explicit retention and privacy rules.
4. Calibrate prediction scores before treating them as probabilities.
5. Add document extraction/OCR as a separate pipeline component.
