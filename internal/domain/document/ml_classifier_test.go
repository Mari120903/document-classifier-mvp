package document

import (
	"fmt"
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

func TestPredictWithFlagsShortCompleteTextIsNotIncomplete(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(validExamples()); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	_, _, flags := nb.PredictWithFlags("Factura total $100")
	if flags.Incomplete {
		t.Fatal("short text without a truncation marker must not be marked incomplete")
	}
}

func TestPredictWithFlagsExplicitTruncationMarker(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(validExamples()); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	_, _, flags := nb.PredictWithFlags("Factura total $100 [truncado]")
	if !flags.Incomplete {
		t.Fatal("expected Incomplete=true when an explicit truncation marker is present")
	}
	if !flags.NeedsReview {
		t.Fatal("expected NeedsReview=true when incomplete")
	}
}

func TestPredictScoresAreNormalizedAndConsistentWithPredict(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(validExamples()); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	text := "Factura total a pagar vencimiento"
	scores := nb.PredictScores(text)

	if len(scores) != len(nb.labels) {
		t.Fatalf("expected scores for %d configured labels, got %d", len(nb.labels), len(scores))
	}

	sum := 0.0
	bestLabel := DocTypeUnknown
	bestScore := 0.0
	for _, label := range nb.labels {
		score, ok := scores[label]
		if !ok {
			t.Errorf("missing score for configured label %s", label)
			continue
		}
		if math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > 1 {
			t.Errorf("score for %s is outside [0, 1]: %v", label, score)
		}
		sum += score
		if score > bestScore {
			bestScore = score
			bestLabel = label
		}
	}

	if math.Abs(sum-1.0) > 1e-12 {
		t.Fatalf("expected scores to sum to 1, got %.16f", sum)
	}
	if bestLabel != DocTypeFinancial {
		t.Fatalf("expected FINANCIAL to have the highest score, got %s", bestLabel)
	}

	predictedLabel, predictedScore := nb.Predict(text)
	if predictedLabel != bestLabel {
		t.Fatalf("Predict() label = %s, highest-scoring label = %s", predictedLabel, bestLabel)
	}
	if math.Abs(predictedScore-bestScore) > 1e-12 {
		t.Fatalf("Predict() score = %.16f, highest score = %.16f", predictedScore, bestScore)
	}
}

func TestPredictScoresBeforeTrainingReturnsZeroForConfiguredLabels(t *testing.T) {
	nb := newTestClassifier(1.0)
	scores := nb.PredictScores("Factura total a pagar")

	if len(scores) != len(nb.labels) {
		t.Fatalf("expected %d configured labels, got %d scores", len(nb.labels), len(scores))
	}
	for _, label := range nb.labels {
		if score, ok := scores[label]; !ok || score != 0 {
			t.Errorf("expected score 0 for %s before training, got %v (present: %t)", label, score, ok)
		}
	}
}

func TestPredictScoresLeavesUntrainedLabelsAtZero(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(validExamples()[:1]); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	scores := nb.PredictScores("Factura total a pagar")
	if scores[DocTypeFinancial] != 1 {
		t.Fatalf("expected trained FINANCIAL label to score 1, got %v", scores[DocTypeFinancial])
	}
	for _, label := range []DocType{DocTypeLegal, DocTypeTextual} {
		if scores[label] != 0 {
			t.Errorf("expected untrained label %s to score 0, got %v", label, scores[label])
		}
	}
}

func TestPredictScoresWithUnusableTextReturnsZeroScores(t *testing.T) {
	nb := newTestClassifier(1.0)
	if err := nb.Train(validExamples()); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	scores := nb.PredictScores("de la y o")
	if len(scores) != len(nb.labels) {
		t.Fatalf("expected %d configured labels, got %d scores", len(nb.labels), len(scores))
	}
	for _, label := range nb.labels {
		if scores[label] != 0 {
			t.Errorf("expected score 0 for unusable input and label %s, got %v", label, scores[label])
		}
	}
}

func totalDocuments(nb *NaiveBayes) int {
	total := 0
	for _, count := range nb.docCounts {
		total += count
	}
	return total
}

func TestTokenizePreservesSpanishAccents(t *testing.T) {
	input := "Cláusula obligación jurisdicción gráfico medición"

	tokens := tokenize(input)

	expected := []string{
		"cláusula",
		"obligación",
		"jurisdicción",
		"gráfico",
		"medición",
	}

	if len(tokens) != len(expected) {
		t.Fatalf("expected %d tokens, got %d: %v",
			len(expected), len(tokens), tokens)
	}

	for i, token := range expected {
		if tokens[i] != token {
			t.Errorf("expected token %q at position %d, got %q",
				token, i, tokens[i])
		}
	}
}
