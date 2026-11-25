FROM golang:1.25.6-alpine AS builder

WORKDIR /src

COPY go.mod ./
COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /app/server ./cmd/api

FROM scratch

COPY --from=builder /app/server /server
COPY --from=builder /src/data/documents.jsonl /data/documents.jsonl

USER 65532:65532

ENV HTTP_ADDR=:8080
ENV DATASET_PATH=/data/documents.jsonl

EXPOSE 8080

ENTRYPOINT ["/server"]
