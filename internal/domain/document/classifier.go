package document

import "strings"

// ClassifyText applies simple MVP rules to determine DocType, confidence and flags.
// This is intentionally rule-based (no LLM yet) to keep behavior testable and predictable.
func ClassifyText(text string) (DocType, float64, Flags) {
	t := strings.TrimSpace(text)
	flags := Flags{}

	// UNREADABLE
	if t == "" {
		flags.Unreadable = true
		flags.NeedsReview = true
		return DocTypeUnknown, 0.0, flags
	}

	// INCOMPLETE (very small heuristic)
	if len([]rune(t)) < 80 {
		flags.Incomplete = true
		flags.NeedsReview = true
	}

	low := strings.ToLower(t)

	// SUSPICIOUS patterns (MVP heuristics)
	if containsAny(low,
		"powershell", "cmd.exe", "curl | sh", "wget ", "bash -c",
		"base64", "<script", "javascript:", "http://", "https://",
	) {
		flags.Suspicious = true
		flags.NeedsReview = true
	}

	// Type classification (very simple keyword scoring)
	financialScore := scoreContains(low, []string{"factura", "invoice", "total", "monto", "vencimiento", "iva", "$", "usd"})
	legalScore := scoreContains(low, []string{"cláusula", "clausula", "contrato", "obligación", "obligacion", "acuerdo", "responsabilidad", "jurisdicción", "jurisdiccion"})
	graphicalScore := scoreContains(low, []string{"gráfico", "grafico", "diagrama", "plano", "esquema", "porcentaje", "%", "barra", "medición", "medicion"})

	// Pick the max
	max := financialScore
	docType := DocTypeFinancial

	if legalScore > max {
		max = legalScore
		docType = DocTypeLegal
	}
	if graphicalScore > max {
		max = graphicalScore
		docType = DocTypeGraphical
	}

	// If no strong signals, fallback
	if max == 0 {
		docType = DocTypeTextual
	}

	// Confidence: simple mapping
	conf := 0.55
	if max >= 3 {
		conf = 0.85
	} else if max == 2 {
		conf = 0.72
	} else if max == 1 {
		conf = 0.62
	}

	// Needs review if low confidence (rule)
	if conf < 0.65 {
		flags.NeedsReview = true
	}

	return docType, conf, flags
}

func containsAny(s string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(s, n) {
			return true
		}
	}
	return false
}

func scoreContains(s string, keywords []string) int {
	score := 0
	for _, k := range keywords {
		if strings.Contains(s, k) {
			score++
		}
	}
	return score
}
