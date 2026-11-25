package dataset

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
)

const maxRecordSize = 1024 * 1024 // 1 MB per JSONL record

// ReadJSONL reads and validates a JSONL dataset.
func ReadJSONL(reader io.Reader) ([]Record, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), maxRecordSize)

	var records []Record

	seenIDs := make(map[string]bool)
	familyLabels := make(map[string]document.DocType)

	for line := 1; scanner.Scan(); line++ {
		raw := strings.TrimSpace(scanner.Text())

		if raw == "" {
			continue
		}

		var record Record

		decoder := json.NewDecoder(bytes.NewReader([]byte(raw)))
		decoder.DisallowUnknownFields()

		if err := decoder.Decode(&record); err != nil {
			return nil, fmt.Errorf("line %d: invalid JSON: %w", line, err)
		}

		if err := validateRecord(record); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}

		record.ID = strings.TrimSpace(record.ID)
		record.Family = strings.TrimSpace(record.Family)

		if seenIDs[record.ID] {
			return nil, fmt.Errorf(
				"line %d: duplicate document ID %q",
				line,
				record.ID,
			)
		}

		if previousLabel, exists := familyLabels[record.Family]; exists {
			if previousLabel != record.Label {
				return nil, fmt.Errorf(
					"line %d: family %q has inconsistent labels: %s and %s",
					line,
					record.Family,
					previousLabel,
					record.Label,
				)
			}
		} else {
			familyLabels[record.Family] = record.Label
		}

		seenIDs[record.ID] = true
		records = append(records, record)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading dataset: %w", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}

	return records, nil
}

func validateRecord(record Record) error {
	if strings.TrimSpace(record.ID) == "" {
		return fmt.Errorf("document ID cannot be empty")
	}

	if strings.TrimSpace(record.Family) == "" {
		return fmt.Errorf("document family cannot be empty")
	}

	switch record.Label {
	case document.DocTypeFinancial,
		document.DocTypeLegal,
		document.DocTypeGraphical,
		document.DocTypeTextual:
	default:
		return fmt.Errorf("unsupported document label %q", record.Label)
	}

	switch record.Difficulty {
	case DifficultyEasy, DifficultyMedium, DifficultyHard:
	default:
		return fmt.Errorf("unsupported difficulty %q", record.Difficulty)
	}

	if strings.TrimSpace(record.Text) == "" {
		return fmt.Errorf("document text cannot be empty")
	}

	return nil
}
