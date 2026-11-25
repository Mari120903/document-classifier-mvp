package dataset

import (
	"strings"
	"testing"

	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
)

func TestReadJSONLValidRecords(t *testing.T) {
	input := `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura de servicios"}
{"id":"doc002","family":"contract","label":"LEGAL","difficulty":"medium","text":"Contrato de locación"}
`

	records, err := ReadJSONL(strings.NewReader(input))

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(records))
	}

	if records[0].Label != document.DocTypeFinancial {
		t.Errorf("unexpected first label: %s", records[0].Label)
	}

	if records[1].Difficulty != DifficultyMedium {
		t.Errorf("unexpected second difficulty: %s", records[1].Difficulty)
	}
}

func TestReadJSONLRejectsInvalidRecords(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "invalid JSON",
			input: `{"id":`,
		},
		{
			name:  "empty ID",
			input: `{"id":"","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura"}`,
		},
		{
			name:  "empty family",
			input: `{"id":"doc001","family":"","label":"FINANCIAL","difficulty":"easy","text":"Factura"}`,
		},
		{
			name:  "unknown label",
			input: `{"id":"doc001","family":"invoice","label":"UNKNOWN","difficulty":"easy","text":"Factura"}`,
		},
		{
			name:  "unknown difficulty",
			input: `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"extreme","text":"Factura"}`,
		},
		{
			name:  "empty text",
			input: `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":" "}`,
		},
		{
			name: "duplicate ID",
			input: `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura"}
{"id":"doc001","family":"receipt","label":"FINANCIAL","difficulty":"easy","text":"Recibo"}`,
		},
		{
			name:  "unknown JSON field",
			input: `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura","extra":"unexpected"}`,
		},
		{
			name: "inconsistent family labels",
			input: `{"id":"doc001","family":"shared","label":"FINANCIAL","difficulty":"easy","text":"Factura"}
{"id":"doc002","family":"shared","label":"LEGAL","difficulty":"medium","text":"Contrato"}`,
		},
		{
			name:  "trailing garbage",
			input: `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura"} basura`,
		},
		{
			name:  "additional JSON object",
			input: `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura"} {"id":"doc002","family":"receipt","label":"FINANCIAL","difficulty":"easy","text":"Recibo"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ReadJSONL(strings.NewReader(tt.input))

			if err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestReadJSONLRejectsEmptyDataset(t *testing.T) {
	_, err := ReadJSONL(strings.NewReader("\n  \n"))

	if err == nil {
		t.Fatal("expected error for empty dataset")
	}
}

func TestReadJSONLAllowsBlankLines(t *testing.T) {
	input := `
{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura"}

`

	records, err := ReadJSONL(strings.NewReader(input))

	if err != nil {
		t.Fatalf("expected blank lines to be ignored, got %v", err)
	}

	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
}

func TestReadJSONLAllowsTrailingWhitespace(t *testing.T) {
	input := `{"id":"doc001","family":"invoice","label":"FINANCIAL","difficulty":"easy","text":"Factura"}   `

	records, err := ReadJSONL(strings.NewReader(input))

	if err != nil {
		t.Fatalf("expected trailing whitespace to be accepted, got %v", err)
	}

	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
}
