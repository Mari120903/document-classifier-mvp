package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/Mari120903/document-classifier-mvp/internal/dataset"
	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
)

var labels = []document.DocType{
	document.DocTypeFinancial,
	document.DocTypeGraphical,
	document.DocTypeLegal,
	document.DocTypeTextual,
}

func main() {
	path := "data/documents.jsonl"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	file, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "opening dataset: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	records, err := dataset.ReadJSONL(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "reading dataset: %v\n", err)
		os.Exit(1)
	}

	train, test, err := splitByFamily(records)
	if err != nil {
		fmt.Fprintf(os.Stderr, "splitting dataset: %v\n", err)
		os.Exit(1)
	}

	examples := make([]document.Example, 0, len(train))
	for _, record := range train {
		examples = append(examples, document.Example{
			Label: record.Label,
			Text:  record.Text,
		})
	}

	model := document.NewNaiveBayes(1.0, labels...)
	if err := model.Train(examples); err != nil {
		fmt.Fprintf(os.Stderr, "training model: %v\n", err)
		os.Exit(1)
	}

	majority := majorityLabel(train)

	fmt.Println("DOCUMENT CLASSIFIER EVALUATION")
	fmt.Println("--------------------------------")
	fmt.Printf("Total records: %d\n", len(records))
	fmt.Printf("Training records: %d\n", len(train))
	fmt.Printf("Test records: %d\n", len(test))
	fmt.Printf("Majority baseline label: %s\n", majority)

	evaluate("NAIVE BAYES", test, func(text string) document.DocType {
		label, _ := model.Predict(text)
		return label
	})

	evaluate("HEURISTIC CLASSIFIER", test, func(text string) document.DocType {
		label, _, _ := document.ClassifyText(text)
		return label
	})

	evaluate("MAJORITY BASELINE", test, func(string) document.DocType {
		return majority
	})
}

// splitByFamily keeps every document from the same family
// in either training or test, never in both.
func splitByFamily(records []dataset.Record) ([]dataset.Record, []dataset.Record, error) {
	grouped := make(map[document.DocType]map[string][]dataset.Record)

	for _, label := range labels {
		grouped[label] = make(map[string][]dataset.Record)
	}

	for _, record := range records {
		grouped[record.Label][record.Family] = append(
			grouped[record.Label][record.Family],
			record,
		)
	}

	rng := rand.New(rand.NewSource(42))

	var train []dataset.Record
	var test []dataset.Record

	for _, label := range labels {
		families := make([]string, 0, len(grouped[label]))

		for family := range grouped[label] {
			families = append(families, family)
		}

		if len(families) < 2 {
			return nil, nil, fmt.Errorf(
				"label %s needs at least 2 families; found %d",
				label,
				len(families),
			)
		}

		sort.Strings(families)
		rng.Shuffle(len(families), func(i, j int) {
			families[i], families[j] = families[j], families[i]
		})

		testCount := int(math.Ceil(float64(len(families)) * 0.20))
		if testCount < 1 {
			testCount = 1
		}
		if testCount >= len(families) {
			testCount = len(families) - 1
		}

		for i, family := range families {
			if i < testCount {
				test = append(test, grouped[label][family]...)
			} else {
				train = append(train, grouped[label][family]...)
			}
		}
	}

	return train, test, nil
}

func majorityLabel(records []dataset.Record) document.DocType {
	counts := make(map[document.DocType]int)

	for _, record := range records {
		counts[record.Label]++
	}

	best := document.DocTypeUnknown
	bestCount := -1

	for _, label := range labels {
		if counts[label] > bestCount {
			best = label
			bestCount = counts[label]
		}
	}

	return best
}

func evaluate(
	name string,
	test []dataset.Record,
	predict func(string) document.DocType,
) {
	confusion := make(map[document.DocType]map[document.DocType]int)

	for _, actual := range labels {
		confusion[actual] = make(map[document.DocType]int)
	}

	correct := 0

	fmt.Println()
	fmt.Println("INDIVIDUAL PREDICTIONS")
	fmt.Printf("%-10s %-25s %-12s %-12s %-10s\n",
		"ID", "FAMILY", "ACTUAL", "PREDICTED", "RESULT")

	for _, record := range test {
		predicted := predict(record.Text)
		confusion[record.Label][predicted]++

		result := "ERROR"
		if predicted == record.Label {
			correct++
			result = "CORRECT"
		}

		fmt.Printf("%-10s %-25s %-12s %-12s %-10s\n",
			record.ID,
			record.Family,
			record.Label,
			predicted,
			result,
		)
	}

	accuracy := float64(correct) / float64(len(test))

	macroPrecision := 0.0
	macroRecall := 0.0
	macroF1 := 0.0

	for _, label := range labels {
		tp := confusion[label][label]
		fp := 0
		fn := confusion[label][document.DocTypeUnknown]

		for _, other := range labels {
			if other != label {
				fp += confusion[other][label]
				fn += confusion[label][other]
			}
		}

		precision := 0.0
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}

		recall := 0.0
		if tp+fn > 0 {
			recall = float64(tp) / float64(tp+fn)
		}

		f1 := 0.0
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}

		macroPrecision += precision
		macroRecall += recall
		macroF1 += f1
	}

	classCount := float64(len(labels))
	macroPrecision /= classCount
	macroRecall /= classCount
	macroF1 /= classCount

	fmt.Println()
	fmt.Println(name)
	fmt.Println("--------------------------------")
	fmt.Printf("Accuracy:         %.2f%%\n", accuracy*100)
	fmt.Printf("Macro precision:  %.2f%%\n", macroPrecision*100)
	fmt.Printf("Macro recall:     %.2f%%\n", macroRecall*100)
	fmt.Printf("Macro F1:         %.2f%%\n", macroF1*100)

	fmt.Println()
	fmt.Println("CONFUSION MATRIX")
	fmt.Printf("%-12s", "ACTUAL/PRED")

	for _, label := range labels {
		fmt.Printf("%-12s", label)
	}
	fmt.Printf("%-12s\n", document.DocTypeUnknown)

	for _, actual := range labels {
		fmt.Printf("%-12s", actual)

		for _, predicted := range labels {
			fmt.Printf("%-12d", confusion[actual][predicted])
		}

		fmt.Printf("%-12d\n", confusion[actual][document.DocTypeUnknown])
	}
}
