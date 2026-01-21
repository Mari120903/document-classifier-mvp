# Document Classification Rules (MVP)

This document defines the basic rules used by the system to
classify documents and assign flags.

The goal is not perfect accuracy, but clear, testable behavior.

---

## 1. Unreadable Document

A document is marked as UNREADABLE when:

- No text can be extracted from the document
- Extracted text is empty or null
- The document format is not supported

Result:
- flag_unreadable = true
- doc_type = UNKNOWN
- confidence = 0.0

---

## 2. Incomplete Document

A document is marked as INCOMPLETE when:

- Extracted text length is below a minimum threshold
- The text ends abruptly or appears cut off
- Required elements for the detected type are missing

Examples:
- A financial document without amount or date
- A legal document without identifiable parties

Result:
- flag_incomplete = true
- flag_needs_review = true

---

## 3. Suspicious Document

A document is marked as SUSPICIOUS when:

- File extension does not match its content type
- The document contains suspicious keywords or patterns
- The document includes excessive encoded or obfuscated text

Examples:
- Embedded commands or scripts
- Multiple suspicious URLs

Result:
- flag_suspicious = true
- flag_needs_review = true

---

## 4. Document Type Classification

Document types are assigned based on detected content patterns:

- FINANCIAL: amounts, dates, invoices, totals
- LEGAL: legal language, clauses, obligations
- GRAPHICAL: references to diagrams, charts, measurements
- TEXTUAL: general written communication
- UNKNOWN: type cannot be confidently determined

---

## 5. Confidence and Review

- Confidence is a value between 0.0 and 1.0
- If confidence is below a defined threshold:
  - flag_needs_review = true
