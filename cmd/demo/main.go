package main

import (
	"fmt"
	"log"

	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
)

func main() {
	nb := document.NewNaiveBayes(1.0,
		document.DocTypeFinancial,
		document.DocTypeLegal,
		document.DocTypeTextual,
		document.DocTypeGraphical,
	)

	err := nb.Train([]document.Example{
		{Label: document.DocTypeFinancial, Text: "Factura total a pagar $32000 vencimiento IVA"},
		{Label: document.DocTypeLegal, Text: "Contrato cláusula jurisdicción obligaciones"},
		{Label: document.DocTypeGraphical, Text: "Plano diagrama esquema porcentaje"},
		{Label: document.DocTypeTextual, Text: "Hola te escribo para coordinar una reunión"},
	})
	if err != nil {
		log.Fatalf("training failed: %v", err)
	}

	samples := []string{
		"Factura N° 123 total $5000 vencimiento",
		"Contrato con cláusulas legales y jurisdicción",
		"Plano con diagrama y mediciones",
		"Hola ¿cómo estás?",
		"",
	}

	for _, sample := range samples {
		label, score, flags := nb.PredictWithFlags(sample)
		fmt.Println("TEXT:", sample)
		fmt.Println("TYPE:", label, "SCORE:", score)
		fmt.Println("FLAGS:", flags)
		fmt.Println("----")
	}
}
