# Document Classifier MVP

## Goal
A system that receives a document, processes it asynchronously,
and classifies its type.

## Document States
- UPLOADED
- PROCESSING
- PROCESSED
- FAILED

## Document Flags
- UNREADABLE
- INCOMPLETE
- SUSPICIOUS
- NEEDS_REVIEW

## Document Types
- TEXTUAL
- LEGAL
- FINANCIAL
- GRAPHICAL
- UNKNOWN

## High-Level Flow
1. A document is uploaded to the system
2. The system processes the document asynchronously
3. Text is extracted from the document
4. The document is classified
5. Status, type, and flags are stored

## Run tests

Requires Go (tested with go1.25.x).

```bash
go test ./...

## Machine Learning (Naive Bayes)

This repo includes an ML-based text classifier (Multinomial Naive Bayes).
It can be trained with labeled examples and produces:
- predicted document type
- confidence score
- review flags

Run demo:

```bash
go run ./cmd/demo
