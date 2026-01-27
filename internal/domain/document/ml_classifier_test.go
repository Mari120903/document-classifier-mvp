package document

import "testing"

func TestNaiveBayesPrediction(t *testing.T) {
	nb := NewNaiveBayes(1.0,
		DocTypeFinancial,
		DocTypeLegal,
		DocTypeTextual,
		DocTypeGraphical,
	)

	nb.Train([]Example{
		{DocTypeFinancial, "Factura total a pagar $32000 vencimiento IVA"},
		{DocTypeLegal, "Contrato cláusula jurisdicción obligaciones"},
		{DocTypeGraphical, "Plano diagrama esquema porcentaje"},
		{DocTypeTextual, "Hola te escribo para coordinar una reunión"},
	})

	dt, conf := nb.Predict("Factura N° 123 total $5000 vencimiento")
	if dt != DocTypeFinancial {
		t.Fatalf("expected FINANCIAL, got %s", dt)
	}
	if conf <= 0.5 {
		t.Fatalf("expected confidence > 0.5, got %v", conf)
	}
}
