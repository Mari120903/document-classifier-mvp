package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Mari120903/document-classifier-mvp/internal/dataset"
	"github.com/Mari120903/document-classifier-mvp/internal/domain/document"
	"github.com/Mari120903/document-classifier-mvp/internal/httpapi"
)

var labels = []document.DocType{
	document.DocTypeFinancial,
	document.DocTypeGraphical,
	document.DocTypeLegal,
	document.DocTypeTextual,
}

func main() {
	datasetPath := os.Getenv("DATASET_PATH")
	if datasetPath == "" {
		datasetPath = "data/documents.jsonl"
	}

	file, err := os.Open(datasetPath)
	if err != nil {
		log.Fatalf("open training dataset: %v", err)
	}
	records, readErr := dataset.ReadJSONL(file)
	closeErr := file.Close()
	if readErr != nil {
		log.Fatalf("read training dataset: %v", readErr)
	}
	if closeErr != nil {
		log.Fatalf("close training dataset: %v", closeErr)
	}

	examples := make([]document.Example, 0, len(records))
	for _, record := range records {
		examples = append(examples, document.Example{
			Label: record.Label,
			Text:  record.Text,
		})
	}

	model := document.NewNaiveBayes(1.0, labels...)
	if err := model.Train(examples); err != nil {
		log.Fatalf("train classifier: %v", err)
	}

	addr := os.Getenv("HTTP_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	server := &http.Server{
		Addr:              addr,
		Handler:           httpapi.NewHandler(model),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-shutdown
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Printf("graceful shutdown failed: %v", err)
		}
	}()

	log.Printf("document classifier API listening on %s (dataset: %s)", addr, datasetPath)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("HTTP server: %v", err)
	}
}
