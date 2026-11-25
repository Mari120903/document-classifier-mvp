package httpapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
)

func testHandler(t *testing.T) http.Handler {
	t.Helper()
	model := document.NewNaiveBayes(1.0,
		document.DocTypeFinancial,
		document.DocTypeLegal,
		document.DocTypeTextual,
		document.DocTypeGraphical,
	)
	err := model.Train([]document.Example{
		{Label: document.DocTypeFinancial, Text: "factura total monto vencimiento iva"},
		{Label: document.DocTypeLegal, Text: "contrato clausula jurisdiccion obligaciones"},
		{Label: document.DocTypeTextual, Text: "mensaje reunion saludo seguimiento"},
		{Label: document.DocTypeGraphical, Text: "grafico diagrama plano esquema porcentaje"},
	})
	if err != nil {
		t.Fatalf("train model: %v", err)
	}
	return NewHandler(model)
}

func TestHealthEndpoint(t *testing.T) {
	recorder := httptest.NewRecorder()
	testHandler(t).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if !strings.Contains(recorder.Body.String(), `"status":"ok"`) {
		t.Fatalf("unexpected response: %s", recorder.Body.String())
	}
}

func TestClassifyEndpointReturnsPredictionAndScores(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/classify",
		strings.NewReader(`{"text":"Factura total a pagar $5000 vencimiento IVA"}`))
	request.Header.Set("Content-Type", "application/json")

	testHandler(t).ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var response classifyResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Label != document.DocTypeFinancial {
		t.Fatalf("label = %s, want FINANCIAL", response.Label)
	}
	if len(response.Scores) != 4 {
		t.Fatalf("got %d scores, want 4", len(response.Scores))
	}
	if response.Score <= 0 || response.Score > 1 {
		t.Fatalf("score out of range: %v", response.Score)
	}
}

func TestClassifyEndpointRejectsInvalidRequests(t *testing.T) {
	tests := []struct {
		name        string
		contentType string
		body        string
		wantStatus  int
	}{
		{name: "wrong content type", contentType: "text/plain", body: `{"text":"hola"}`, wantStatus: http.StatusUnsupportedMediaType},
		{name: "malformed JSON", contentType: "application/json", body: `{"text":`, wantStatus: http.StatusBadRequest},
		{name: "unknown field", contentType: "application/json", body: `{"text":"hola","extra":true}`, wantStatus: http.StatusBadRequest},
		{name: "multiple JSON values", contentType: "application/json", body: `{"text":"hola"} {"text":"otro"}`, wantStatus: http.StatusBadRequest},
		{name: "non object", contentType: "application/json", body: `[]`, wantStatus: http.StatusBadRequest},
		{name: "missing text field", contentType: "application/json", body: `{}`, wantStatus: http.StatusBadRequest},
		{name: "null text field", contentType: "application/json", body: `{"text":null}`, wantStatus: http.StatusBadRequest},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/v1/classify", strings.NewReader(test.body))
			request.Header.Set("Content-Type", test.contentType)
			testHandler(t).ServeHTTP(recorder, request)
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body = %s", recorder.Code, test.wantStatus, recorder.Body.String())
			}
		})
	}
}

func TestClassifyEndpointAcceptsEmptyTextForReview(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/classify", strings.NewReader(`{"text":""}`))
	request.Header.Set("Content-Type", "application/json")

	testHandler(t).ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body = %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var response classifyResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !response.Flags.Unreadable || !response.Flags.NeedsReview {
		t.Fatalf("flags = %+v, want unreadable and needs_review", response.Flags)
	}
}

func TestClassifyEndpointRejectsOversizedBody(t *testing.T) {
	body := `{"text":"` + strings.Repeat("a", maxRequestBodyBytes) + `"}`
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/classify", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")

	testHandler(t).ServeHTTP(recorder, request)

	if recorder.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusRequestEntityTooLarge)
	}
}

func TestClassifyEndpointUnavailableWithoutModel(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/classify", strings.NewReader(`{"text":"hola"}`))
	request.Header.Set("Content-Type", "application/json")

	NewHandler(nil).ServeHTTP(recorder, request)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusServiceUnavailable)
	}
}
