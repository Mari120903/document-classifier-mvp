# Document Processing Flow (MVP)

This document describes how a document moves through the system
from upload to final classification.

---

## 1. Upload

- A document is received by the system.
- A new Document record is created.
- Initial status is set to UPLOADED.

Possible issues:
- Invalid or unsupported format
- Empty file

---

## 2. Processing Start

- The system starts processing the document asynchronously.
- Status changes from UPLOADED to PROCESSING.

Possible issues:
- Processing service unavailable
- Timeout while starting processing

---

## 3. Text Extraction

- The system attempts to extract text from the document.
- If no text can be extracted:
  - flag_unreadable = true
  - status remains PROCESSING or moves to FAILED (depending on severity)

Possible issues:
- Corrupted document
- Unsupported encoding

---

## 4. Classification

- Extracted text is analyzed.
- Document type is determined.
- Confidence score is assigned.
- Flags are set according to defined rules.

Possible issues:
- Ambiguous content
- Low confidence classification

---

## 5. Completion

- Status is set to PROCESSED.
- Classification result and flags are stored.
- Document is available for review or further actions.

---

## 6. Failure Handling

- If a critical error occurs at any step:
  - Status is set to FAILED.
  - Error information is stored for debugging.
