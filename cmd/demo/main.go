package main

import (
	"fmt"

	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
)

func main() {
	nb := document.NewNaiveBayes(1.0,
		document.DocTypeFinancial,
		document.DocTypeLegal,
		document.DocTypeTextual,
		document.DocTypeGraphical,
	)

	nb.Train([]document.Example{
		{document.DocTypeFinancial, "Factura total a pagar $32000 vencimiento IVA"},
		{document.DocTypeLegal, "Contrato cláusula jurisdicción obligaciones"},
		{document.DocTypeGraphical, "Plano diagrama esquema porcentaje"},
		{document.DocTypeTextual, "Hola te escribo para coordinar una reunión"},
	})

	samples := []string{
		"Factura N° 123 total $5000 vencimiento",
		"Contrato con cláusulas legales y jurisdicción",
		"Plano con diagrama y mediciones",
		"Hola ¿cómo estás?",
		"",
	}

	for _, s := range samples {
		dt, conf, flags := nb.PredictWithFlags(s)
		fmt.Println("TEXT:", s)
		fmt.Println("TYPE:", dt, "CONF:", conf)
		fmt.Println("FLAGS:", flags)
		fmt.Println("----")
	}
}
