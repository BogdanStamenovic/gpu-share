package broker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Ollama struct {
	Base      string
	ModelsDir string
	HTTP      *http.Client
}

type PSModel struct {
	Name     string `json:"name"`
	Model    string `json:"model"`
	Size     int64  `json:"size"`
	SizeVRAM int64  `json:"size_vram"`
	Ctx      int    `json:"context_length"`
}

func (o *Ollama) PS(ctx context.Context) ([]PSModel, error) {
	var r struct {
		Models []PSModel `json:"models"`
	}
	if err := o.getJSON(ctx, "/api/ps", &r); err != nil {
		return nil, err
	}
	return r.Models, nil
}

// Unload asks ollama to drop a model now (keep_alive 0), the documented way.
func (o *Ollama) Unload(ctx context.Context, model string) error {
	body, _ := json.Marshal(map[string]any{"model": model, "keep_alive": 0})
	req, _ := http.NewRequestWithContext(ctx, "POST", o.Base+"/api/generate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.HTTP.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		// bge-m3 and other embedding models reject /api/generate; /api/embed accepts keep_alive.
		body, _ = json.Marshal(map[string]any{"model": model, "input": "", "keep_alive": 0})
		req, _ = http.NewRequestWithContext(ctx, "POST", o.Base+"/api/embed", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp, err = o.HTTP.Do(req)
		if err != nil {
			return err
		}
		resp.Body.Close()
	}
	return nil
}

func (o *Ollama) getJSON(ctx context.Context, path string, v any) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", o.Base+path, nil)
	resp, err := o.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("%s: HTTP %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

type manifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

// manifestPaths lists where a model's manifest can live, newest layout first.
// Ollama 0.40 writes new manifests under manifests-v2/ollama.com/... but does
// not migrate models pulled earlier, which stay under
// manifests/registry.ollama.ai/... Both layouts are live in the same store.
func (o *Ollama) manifestPaths(model string) []string {
	name, tag, ok := strings.Cut(model, ":")
	if !ok {
		tag = "latest"
	}
	parts := strings.Split(name, "/")
	if len(parts) == 1 {
		parts = []string{"library", parts[0]}
	}
	join := func(root, host string) string {
		p := []string{o.ModelsDir, root}
		if host != "" {
			p = append(p, host)
		}
		return filepath.Join(append(append(p, parts...), tag)...)
	}
	if len(parts) == 2 {
		return []string{join("manifests-v2", "ollama.com"), join("manifests", "registry.ollama.ai")}
	}
	return []string{join("manifests-v2", ""), join("manifests", "")} // host is in the name (hf.co/...)
}

func (o *Ollama) readManifest(model string) (manifest, error) {
	var m manifest
	var err error
	for _, p := range o.manifestPaths(model) {
		var b []byte
		if b, err = os.ReadFile(p); err == nil {
			return m, json.Unmarshal(b, &m)
		}
	}
	return m, err
}

func blobPath(dir, digest string) string {
	return filepath.Join(dir, "blobs", strings.Replace(digest, ":", "-", 1))
}

// ResolveGGUF finds a model's weights through its manifest.
func (o *Ollama) ResolveGGUF(model string) (string, error) {
	m, err := o.readManifest(model)
	if err != nil {
		return "", fmt.Errorf("manifest for %s: %w", model, err)
	}
	for _, l := range m.Layers {
		if l.MediaType == "application/vnd.ollama.image.model" {
			return blobPath(o.ModelsDir, l.Digest), nil
		}
	}
	return "", fmt.Errorf("%s: no model layer in manifest", model)
}

// WeightsMiB is the size of a model's weights on disk, used as a lower bound
// for a model ollama has never loaded while we were watching.
func (o *Ollama) WeightsMiB(model string) int {
	m, err := o.readManifest(model)
	if err != nil {
		return 0
	}
	var n int64
	for _, l := range m.Layers {
		if l.MediaType == "application/vnd.ollama.image.model" || l.MediaType == "application/vnd.ollama.image.projector" {
			n += l.Size
		}
	}
	return int(n >> 20)
}

// Params returns the model's ollama default options (temperature etc.), so the
// llama-server serving it samples the way ollama did when a client sends none.
func (o *Ollama) Params(model string) map[string]any {
	m, err := o.readManifest(model)
	if err != nil {
		return nil
	}
	for _, l := range m.Layers {
		if l.MediaType == "application/vnd.ollama.image.params" {
			b, err := os.ReadFile(blobPath(o.ModelsDir, l.Digest))
			if err != nil {
				return nil
			}
			var p map[string]any
			if json.Unmarshal(b, &p) == nil {
				return p
			}
		}
	}
	return nil
}
