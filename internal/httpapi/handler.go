package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"

	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
)

const maxRequestBodyBytes = 1 << 20

type classifyRequest struct {
	Text *string `json:"text"`
}

type classifyResponse struct {
	Label  document.DocType             `json:"label"`
	Score  float64                      `json:"score"`
	Scores map[document.DocType]float64 `json:"scores"`
	Flags  document.Flags               `json:"flags"`
}

type errorResponse struct {
	Error string `json:"error"`
}

type healthResponse struct {
	Status string `json:"status"`
}

type handler struct {
	model *document.NaiveBayes
}

// NewHandler creates the HTTP API handler for a trained Naive Bayes model.
func NewHandler(model *document.NaiveBayes) http.Handler {
	mux := http.NewServeMux()
	h := &handler{model: model}

	mux.HandleFunc("GET /healthz", h.health)
	mux.HandleFunc("POST /v1/classify", h.classify)

	return mux
}

func (h *handler) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, healthResponse{Status: "ok"})
}

func (h *handler) classify(w http.ResponseWriter, r *http.Request) {
	if h.model == nil {
		writeError(w, http.StatusServiceUnavailable, "classifier is not initialized")
		return
	}

	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		writeError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return
	}

	body := http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	defer body.Close()

	payload, err := io.ReadAll(body)
	if err != nil {
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body exceeds 1 MiB")
			return
		}
		writeError(w, http.StatusBadRequest, "could not read request body")
		return
	}

	trimmed := bytes.TrimSpace(payload)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		writeError(w, http.StatusBadRequest, "request body must be a JSON object")
		return
	}

	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.DisallowUnknownFields()

	var request classifyRequest
	if err := decoder.Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request JSON: "+err.Error())
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		writeError(w, http.StatusBadRequest, "request body must contain exactly one JSON object")
		return
	}
	if request.Text == nil {
		writeError(w, http.StatusBadRequest, "field text is required and must be a string")
		return
	}

	text := *request.Text
	label, score, flags := h.model.PredictWithFlags(text)
	_, _, heuristicFlags := document.ClassifyText(text)
	flags.Suspicious = heuristicFlags.Suspicious
	if flags.Suspicious {
		flags.NeedsReview = true
	}

	writeJSON(w, http.StatusOK, classifyResponse{
		Label:  label,
		Score:  score,
		Scores: h.model.PredictScores(text),
		Flags:  flags,
	})
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		// The response has already been committed; there is no safe recovery action.
		return
	}
}
