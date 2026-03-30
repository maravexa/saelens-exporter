// Package saelens provides a Go client for the saelens-exporter
// Unix domain socket protocol.
//
// This is a reference implementation for integrating into the
// ollama-exporter Go control plane. The Go side owns the Prometheus
// /metrics endpoint and proxies SAE displacement metrics scraped
// from the textfile collector directory.
//
// Usage in your Go exporter:
//
//	client := saelens.NewClient("/tmp/saelens-exporter.sock")
//	resp, err := client.Scan(ctx, []string{"prompt1", "prompt2"}, "garak-scan-001")
//	if err != nil { ... }
//	for _, r := range resp.Results {
//	    if r.Alert { log.Warn("displacement detected", "probe", r.ProbeID) }
//	}
package saelens

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"time"
)

const (
	maxMessageSize = 16 * 1024 * 1024 // 16MB
	dialTimeout    = 5 * time.Second
)

// Client communicates with the saelens-exporter Python worker
// over a Unix domain socket using length-prefixed JSON framing.
type Client struct {
	socketPath string
}

func NewClient(socketPath string) *Client {
	return &Client{socketPath: socketPath}
}

// ScanRequest is sent to the Python worker.
type ScanRequest struct {
	Command string   `json:"command"`
	Prompts []string `json:"prompts"`
	ScanID  string   `json:"scan_id"`
}

// ScanResponse comes back from the Python worker.
type ScanResponse struct {
	Status       string       `json:"status"`
	ScanID       string       `json:"scan_id"`
	TotalPrompts int          `json:"total_prompts"`
	AlertCount   int          `json:"alert_count"`
	Results      []ProbeResult `json:"results"`
	Message      string       `json:"message,omitempty"` // error case
}

// ProbeResult is the per-prompt displacement result.
type ProbeResult struct {
	ProbeID        string        `json:"probe_id"`
	Prompt         string        `json:"prompt"`
	Displacement   float64       `json:"displacement"`
	ActiveFeatures int           `json:"active_features"`
	Alert          bool          `json:"alert"`
	DurationS      float64       `json:"duration_s"`
	TopFeatures    []FeatureHit  `json:"top_features"`
}

type FeatureHit struct {
	Index int     `json:"index"`
	Value float64 `json:"value"`
}

// Scan sends prompts to the Python worker for displacement analysis.
func (c *Client) Scan(ctx context.Context, prompts []string, scanID string) (*ScanResponse, error) {
	req := ScanRequest{
		Command: "scan",
		Prompts: prompts,
		ScanID:  scanID,
	}
	var resp ScanResponse
	if err := c.call(ctx, req, &resp); err != nil {
		return nil, fmt.Errorf("scan failed: %w", err)
	}
	if resp.Status == "error" {
		return nil, fmt.Errorf("worker error: %s", resp.Message)
	}
	return &resp, nil
}

// Health checks liveness of the Python worker.
func (c *Client) Health(ctx context.Context) error {
	req := map[string]string{"command": "health"}
	var resp map[string]any
	return c.call(ctx, req, &resp)
}

func (c *Client) call(ctx context.Context, req any, resp any) error {
	d := net.Dialer{Timeout: dialTimeout}
	conn, err := d.DialContext(ctx, "unix", c.socketPath)
	if err != nil {
		return fmt.Errorf("dial %s: %w", c.socketPath, err)
	}
	defer conn.Close()

	// Send length-prefixed JSON
	payload, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(payload)))
	if _, err := conn.Write(append(header, payload...)); err != nil {
		return fmt.Errorf("write: %w", err)
	}

	// Read response
	if _, err := conn.Read(header); err != nil {
		return fmt.Errorf("read header: %w", err)
	}
	length := binary.BigEndian.Uint32(header)
	if length > maxMessageSize {
		return fmt.Errorf("response too large: %d bytes", length)
	}

	buf := make([]byte, length)
	n := 0
	for n < int(length) {
		read, err := conn.Read(buf[n:])
		if err != nil {
			return fmt.Errorf("read body: %w", err)
		}
		n += read
	}

	return json.Unmarshal(buf[:n], resp)
}
