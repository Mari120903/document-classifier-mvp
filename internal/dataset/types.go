package dataset

import "github.com/Mari120903/document-classifier-mvp/internal/domain/document"

// Difficulty represents the complexity of a dataset record.
type Difficulty string

const (
	DifficultyEasy   Difficulty = "easy"
	DifficultyMedium Difficulty = "medium"
	DifficultyHard   Difficulty = "hard"
)

// Record represents one labeled document in the dataset.
type Record struct {
	ID         string           `json:"id"`
	Family     string           `json:"family"`
	Label      document.DocType `json:"label"`
	Difficulty Difficulty       `json:"difficulty"`
	Text       string           `json:"text"`
}
