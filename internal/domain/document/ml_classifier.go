package document

import (
	"math"
	"regexp"
	"sort"
	"strings"
)

// NaiveBayes is a Multinomial Naive Bayes classifier for text.
// This is real machine learning: it learns from labeled examples.
type NaiveBayes struct {
	labels []DocType

	tokenCounts map[DocType]map[string]int
	totalTokens map[DocType]int
	docCounts   map[DocType]int
	vocab       map[string]bool

	alpha float64
}

func NewNaiveBayes(alpha float64, labels ...DocType) *NaiveBayes {
	unique := make([]DocType, 0)
	seen := map[DocType]bool{}
	for _, l := range labels {
		if !seen[l] {
			seen[l] = true
			unique = append(unique, l)
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

	for _, l := range unique {
		nb.tokenCounts[l] = map[string]int{}
		nb.totalTokens[l] = 0
		nb.docCounts[l] = 0
	}

	return nb
}

type Example struct {
	Label DocType
	Text  string
}

func (nb *NaiveBayes) Train(examples []Example) {
	for _, ex := range examples {
		tokens := tokenize(ex.Text)
		nb.docCounts[ex.Label]++

		for _, tok := range tokens {
			nb.vocab[tok] = true
			nb.tokenCounts[ex.Label][tok]++
			nb.totalTokens[ex.Label]++
		}
	}
}

func (nb *NaiveBayes) Predict(text string) (DocType, float64) {
	tokens := tokenize(text)
	if len(tokens) == 0 {
		return DocTypeUnknown, 0.0
	}

	totalDocs := 0
	for _, l := range nb.labels {
		totalDocs += nb.docCounts[l]
	}
	if totalDocs == 0 {
		return DocTypeUnknown, 0.0
	}

	vocabSize := float64(len(nb.vocab))
	alpha := nb.alpha

	logScores := map[DocType]float64{}
	bestLabel := DocTypeUnknown
	bestScore := math.Inf(-1)

	for _, label := range nb.labels {
		prior := float64(nb.docCounts[label]) / float64(totalDocs)
		if prior == 0 {
			logScores[label] = math.Inf(-1)
			continue
		}

		score := math.Log(prior)
		denom := float64(nb.totalTokens[label]) + alpha*vocabSize

		for _, tok := range tokens {
			count := float64(nb.tokenCounts[label][tok])
			score += math.Log((count + alpha) / denom)
		}

		logScores[label] = score
		if score > bestScore {
			bestScore = score
			bestLabel = label
		}
	}

	conf := softmax(bestLabel, logScores)
	return bestLabel, conf
}

func (nb *NaiveBayes) PredictWithFlags(text string) (DocType, float64, Flags) {
	t := strings.TrimSpace(text)
	flags := Flags{}

	if t == "" {
		flags.Unreadable = true
		flags.NeedsReview = true
		return DocTypeUnknown, 0.0, flags
	}

	label, conf := nb.Predict(t)

	if conf < 0.65 {
		flags.NeedsReview = true
	}
	if len([]rune(t)) < 80 {
		flags.Incomplete = true
		flags.NeedsReview = true
	}

	return label, conf, flags
}

func softmax(best DocType, logScores map[DocType]float64) float64 {
	max := math.Inf(-1)
	for _, v := range logScores {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	bestVal := 0.0

	for l, v := range logScores {
		if math.IsInf(v, -1) {
			continue
		}
		ev := math.Exp(v - max)
		sum += ev
		if l == best {
			bestVal = ev
		}
	}
	if sum == 0 {
		return 0.0
	}
	return bestVal / sum
}

var nonAlphaNum = regexp.MustCompile(`[^a-z0-9áéíóúñü%$]+`)

func tokenize(text string) []string {
	s := strings.ToLower(strings.TrimSpace(text))
	if s == "" {
		return nil
	}

	s = nonAlphaNum.ReplaceAllString(s, " ")
	parts := strings.Fields(s)

	stop := map[string]bool{
		"de": true, "la": true, "el": true, "y": true, "o": true,
		"the": true, "and": true, "of": true, "to": true,
	}

	var tokens []string
	for _, p := range parts {
		if len([]rune(p)) <= 1 || stop[p] {
			continue
		}
		tokens = append(tokens, p)
	}
	return tokens
}
