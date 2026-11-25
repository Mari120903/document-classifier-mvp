package document

import (
	"errors"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// NaiveBayes is a Multinomial Naive Bayes classifier for text.
// It learns token frequencies from labeled examples.
type NaiveBayes struct {
	labels []DocType

	tokenCounts map[DocType]map[string]int
	totalTokens map[DocType]int
	docCounts   map[DocType]int
	vocab       map[string]bool

	alpha float64
}

type Example struct {
	Label DocType
	Text  string
}

// NewNaiveBayes creates a classifier. Configuration is validated by Train,
// which returns an error if alpha or the label set is invalid.
func NewNaiveBayes(alpha float64, labels ...DocType) *NaiveBayes {
	unique := make([]DocType, 0, len(labels))
	seen := map[DocType]bool{}

	for _, label := range labels {
		if !seen[label] {
			seen[label] = true
			unique = append(unique, label)
		}
	}

	sort.Slice(unique, func(i, j int) bool { return unique[i] < unique[j] })

	nb := &NaiveBayes{
		labels:      unique,
		tokenCounts: map[DocType]map[string]int{},
		totalTokens: map[DocType]int{},
		docCounts:   map[DocType]int{},
		vocab:       map[string]bool{},
		alpha:       alpha,
	}

	for _, label := range unique {
		nb.tokenCounts[label] = map[string]int{}
		nb.totalTokens[label] = 0
		nb.docCounts[label] = 0
	}

	return nb
}

// Train validates the complete batch before changing the model.
// If validation fails, the model remains unchanged.
func (nb *NaiveBayes) Train(examples []Example) error {
	if nb == nil {
		return errors.New("naive bayes classifier is nil")
	}
	if math.IsNaN(nb.alpha) || math.IsInf(nb.alpha, 0) || nb.alpha <= 0 {
		return errors.New("alpha must be a finite number greater than zero")
	}
	if len(nb.labels) == 0 {
		return errors.New("at least one document label must be configured")
	}
	if len(examples) == 0 {
		return errors.New("training examples cannot be empty")
	}

	knownLabels := make(map[DocType]bool, len(nb.labels))
	for _, label := range nb.labels {
		knownLabels[label] = true
	}

	// Build the batch separately so invalid input cannot partially train the model.
	batchTokenCounts := make(map[DocType]map[string]int, len(nb.labels))
	batchTotalTokens := make(map[DocType]int, len(nb.labels))
	batchDocCounts := make(map[DocType]int, len(nb.labels))
	batchVocab := make(map[string]bool)

	for _, label := range nb.labels {
		batchTokenCounts[label] = map[string]int{}
	}

	for i, example := range examples {
		if !knownLabels[example.Label] {
			return errors.New("training example contains an unknown label at index " + strconv.Itoa(i))
		}

		tokens := tokenize(example.Text)
		if len(tokens) == 0 {
			return errors.New("training example has no usable tokens at index " + strconv.Itoa(i))
		}

		batchDocCounts[example.Label]++
		for _, token := range tokens {
			batchVocab[token] = true
			batchTokenCounts[example.Label][token]++
			batchTotalTokens[example.Label]++
		}
	}

	for _, label := range nb.labels {
		nb.docCounts[label] += batchDocCounts[label]
		nb.totalTokens[label] += batchTotalTokens[label]
		for token, count := range batchTokenCounts[label] {
			nb.tokenCounts[label][token] += count
		}
	}
	for token := range batchVocab {
		nb.vocab[token] = true
	}

	return nil
}

// Predict returns the most likely label and its normalized model score.
// The score is not a calibrated probability.
func (nb *NaiveBayes) Predict(text string) (DocType, float64) {
	if nb == nil || len(nb.labels) == 0 || len(nb.vocab) == 0 ||
		math.IsNaN(nb.alpha) || math.IsInf(nb.alpha, 0) || nb.alpha <= 0 {
		return DocTypeUnknown, 0.0
	}

	tokens := tokenize(text)
	if len(tokens) == 0 {
		return DocTypeUnknown, 0.0
	}

	totalDocs := 0
	for _, label := range nb.labels {
		totalDocs += nb.docCounts[label]
	}
	if totalDocs == 0 {
		return DocTypeUnknown, 0.0
	}

	vocabSize := float64(len(nb.vocab))
	logScores := map[DocType]float64{}
	bestLabel := DocTypeUnknown
	bestScore := math.Inf(-1)

	for _, label := range nb.labels {
		if nb.docCounts[label] == 0 {
			logScores[label] = math.Inf(-1)
			continue
		}

		prior := float64(nb.docCounts[label]) / float64(totalDocs)
		score := math.Log(prior)
		denom := float64(nb.totalTokens[label]) + nb.alpha*vocabSize

		for _, token := range tokens {
			count := float64(nb.tokenCounts[label][token])
			score += math.Log((count + nb.alpha) / denom)
		}

		logScores[label] = score
		if score > bestScore {
			bestScore = score
			bestLabel = label
		}
	}

	return bestLabel, softmax(bestLabel, logScores)
}

func (nb *NaiveBayes) PredictWithFlags(text string) (DocType, float64, Flags) {
	trimmed := strings.TrimSpace(text)
	flags := Flags{}

	if trimmed == "" {
		flags.Unreadable = true
		flags.NeedsReview = true
		return DocTypeUnknown, 0.0, flags
	}

	label, score := nb.Predict(trimmed)
	if score < 0.65 {
		flags.NeedsReview = true
	}
	if len([]rune(trimmed)) < 80 {
		flags.Incomplete = true
		flags.NeedsReview = true
	}

	return label, score, flags
}

func softmax(best DocType, logScores map[DocType]float64) float64 {
	maxScore := math.Inf(-1)
	for _, score := range logScores {
		if score > maxScore {
			maxScore = score
		}
	}
	if math.IsInf(maxScore, -1) {
		return 0.0
	}

	sum := 0.0
	bestValue := 0.0
	for label, score := range logScores {
		if math.IsInf(score, -1) {
			continue
		}
		value := math.Exp(score - maxScore)
		sum += value
		if label == best {
			bestValue = value
		}
	}
	if sum == 0 {
		return 0.0
	}
	return bestValue / sum
}

var nonAlphaNum = regexp.MustCompile(`[^a-z0-9áéíóúñü%$]+`)

func tokenize(text string) []string {
	text = strings.ToLower(strings.TrimSpace(text))
	if text == "" {
		return nil
	}

	text = nonAlphaNum.ReplaceAllString(text, " ")
	parts := strings.Fields(text)

	stopWords := map[string]bool{
		"de": true, "la": true, "el": true, "y": true, "o": true,
		"the": true, "and": true, "of": true, "to": true,
	}

	var tokens []string
	for _, part := range parts {
		if len([]rune(part)) <= 1 || stopWords[part] {
			continue
		}
		tokens = append(tokens, part)
	}
	return tokens
}
