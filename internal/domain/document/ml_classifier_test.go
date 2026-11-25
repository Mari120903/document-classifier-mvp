package document

import (
	"math"
	"testing"
)

func newTestClassifier(alpha float64) *NaiveBayes {
	return NewNaiveBayes(alpha, DocTypeFinancial, DocTypeLegal, DocTypeTextual)
}

func validExamples() []Example {
	return []Example{
		{Label: DocTypeFinancial, Text: "Factura total a pagar vencimiento IVA"},
		{Label: DocTypeLegal, Text: "Contrato cláusula jurisdicción obligaciones"},
		{Label: DocTypeTextual, Text: "Hola te escribo para coordinar reunión"},
	}
}

func TestNaiveBayesPrediction(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(validExamples()); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	label, score := nb.Predict("Factura total a pagar vencimiento")
	if label != DocTypeFinancial {
		t.Fatalf("expected FINANCIAL, got %s", label)
	}
	if score <= 0.5 || score > 1 {
		t.Fatalf("expected normalized score in (0.5, 1], got %v", score)
	}
}

func TestTrainRejectsEmptyBatch(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(nil); err == nil {
		t.Fatal("expected error for empty training batch")
	}
}

func TestTrainRejectsUnknownLabel(t *testing.T) {
	nb := newTestClassifier(1.0)
	examples := validExamples()
	examples[1].Label = DocType("UNKNOWN_CUSTOM")

	if err := nb.Train(examples); err == nil {
		t.Fatal("expected error for unknown label")
	}
	if totalDocuments(nb) != 0 {
		t.Fatal("model changed after invalid training batch")
	}
}

func TestTrainRejectsTextWithoutUsableTokens(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train([]Example{
		{Label: DocTypeFinancial, Text: "de la y o"},
	}); err == nil {
		t.Fatal("expected error for text without usable tokens")
	}
	if totalDocuments(nb) != 0 {
		t.Fatal("model changed after invalid training batch")
	}
}

func TestTrainRejectsInvalidAlpha(t *testing.T) {
	values := []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)}
	for _, alpha := range values {
		t.Run(fmt.Sprint(alpha), func(t *testing.T) {
			nb := newTestClassifier(alpha)
			if err := nb.Train(validExamples()); err == nil {
				t.Fatalf("expected error for alpha %v", alpha)
			}
		})
	}
}

func TestTrainRejectsMissingLabelsConfiguration(t *testing.T) {
	nb := NewNaiveBayes(1.0)
	if err := nb.Train(validExamples()); err == nil {
		t.Fatal("expected error when no labels are configured")
	}
}

func TestTrainIsTransactional(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(validExamples()[:1]); err != nil {
		t.Fatalf("initial Train() error = %v", err)
	}

	beforeDocuments := totalDocuments(nb)
	beforeVocabulary := len(nb.vocab)

	invalidBatch := []Example{
		{Label: DocTypeLegal, Text: "contrato válido"},
		{Label: DocType("NOT_CONFIGURED"), Text: "texto inválido"},
	}
	if err := nb.Train(invalidBatch); err == nil {
		t.Fatal("expected error for invalid batch")
	}

	if totalDocuments(nb) != beforeDocuments {
		t.Fatal("document counts changed after rejected batch")
	}
	if len(nb.vocab) != beforeVocabulary {
		t.Fatal("vocabulary changed after rejected batch")
	}
}

func TestPredictBeforeTrainingReturnsUnknown(t *testing.T) {
	nb := newTestClassifier(1.0)
	label, score := nb.Predict("Factura total a pagar")
	if label != DocTypeUnknown || score != 0 {
		t.Fatalf("expected UNKNOWN and zero score, got %s and %v", label, score)
	}
}

func TestPredictWithFlagsOnEmptyText(t *testing.T) {
	nb := newTestClassifier(1.0)
	label, score, flags := nb.PredictWithFlags("   ")
	if label != DocTypeUnknown || score != 0 {
		t.Fatalf("expected UNKNOWN and zero score, got %s and %v", label, score)
	}
	if !flags.Unreadable || !flags.NeedsReview {
		t.Fatalf("expected unreadable and needs-review flags, got %+v", flags)
	}
}

func totalDocuments(nb *NaiveBayes) int {
	total := 0
	for _, count := range nb.docCounts {
		total += count
	}
	return total
}
