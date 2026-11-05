package broker

import (
	"os"
	"path/filepath"
	"testing"
)

func writeManifest(t *testing.T, path, digest string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	b := `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"` + digest + `","size":1}]}`
	if err := os.WriteFile(path, []byte(b), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestResolveGGUFBothManifestLayouts(t *testing.T) {
	dir := t.TempDir()
	o := &Ollama{ModelsDir: dir}
	writeManifest(t, filepath.Join(dir, "manifests/registry.ollama.ai/library/old/8b"), "sha256:aa")
	writeManifest(t, filepath.Join(dir, "manifests-v2/ollama.com/library/new/9b"), "sha256:bb")
	writeManifest(t, filepath.Join(dir, "manifests/hf.co/Org/repo/latest"), "sha256:cc")
	writeManifest(t, filepath.Join(dir, "manifests-v2/ollama.com/user/m/latest"), "sha256:ee")
	// Same name in both layouts: v2 is what current ollama writes, so it wins.
	writeManifest(t, filepath.Join(dir, "manifests/registry.ollama.ai/library/both/1b"), "sha256:d0")
	writeManifest(t, filepath.Join(dir, "manifests-v2/ollama.com/library/both/1b"), "sha256:d1")

	for model, want := range map[string]string{
		"old:8b": "sha256-aa", "new:9b": "sha256-bb", "hf.co/Org/repo": "sha256-cc",
		"user/m": "sha256-ee", "both:1b": "sha256-d1",
	} {
		got, err := o.ResolveGGUF(model)
		if err != nil {
			t.Fatalf("%s: %v", model, err)
		}
		if got != filepath.Join(dir, "blobs", want) {
			t.Errorf("%s: got %s, want blob %s", model, got, want)
		}
	}
	if _, err := o.ResolveGGUF("missing:1b"); err == nil {
		t.Error("missing model resolved")
	}
}
