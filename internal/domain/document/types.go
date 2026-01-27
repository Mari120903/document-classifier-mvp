package document

// Status represents the processing state of a document (one at a time).
type Status string

const (
	StatusUploaded   Status = "UPLOADED"
	StatusProcessing Status = "PROCESSING"
	StatusProcessed  Status = "PROCESSED"
	StatusFailed     Status = "FAILED"
)

// DocType represents the detected type of the document.
type DocType string

const (
	DocTypeTextual   DocType = "TEXTUAL"
	DocTypeLegal     DocType = "LEGAL"
	DocTypeFinancial DocType = "FINANCIAL"
	DocTypeGraphical DocType = "GRAPHICAL"
	DocTypeUnknown   DocType = "UNKNOWN"
)

// Flags are independent signals (can be combined).
type Flags struct {
	Unreadable  bool `json:"unreadable"`
	Incomplete  bool `json:"incomplete"`
	Suspicious  bool `json:"suspicious"`
	NeedsReview bool `json:"needs_review"`
}
