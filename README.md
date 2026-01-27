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
