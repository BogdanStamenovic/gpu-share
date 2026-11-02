package broker

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Profile is one way of serving a model through the turbo3 llama-server.
// Switching profiles means restarting llama-server, because KV type and
// context size are fixed at load time.
type Profile struct {
	// Ollama model whose GGUF blob is served, resolved through ollama's
	// manifest (never a remembered hash: two models here share a family and
	// an earlier session measured the wrong blob).
	OllamaModel string   `json:"ollama_model"`
	GGUF        string   `json:"gguf,omitempty"` // explicit path; overrides OllamaModel
	Ctx         int      `json:"ctx"`
	CacheK      string   `json:"cache_k"`
	CacheV      string   `json:"cache_v"`
	MiB         int      `json:"mib"` // measured VRAM footprint once loaded
	ExtraArgs   []string `json:"extra_args,omitempty"`
}

type Config struct {
	Listen     string `json:"listen"`
	Ollama     string `json:"ollama"`
	MarginMiB  int    `json:"margin_mib"`
	StateDir   string `json:"state_dir"`
	LoadWaitS  int    `json:"load_wait_s"`
	HoldS      int    `json:"hold_s"` // how long an LLM load waits for a call to end before 503
	ReconcileS int    `json:"reconcile_s"`

	LlamaServer string `json:"llama_server"`
	TurboPort   int    `json:"turbo_port"`
	KeepAliveS  int    `json:"keep_alive_s"`

	// Ollama models directory, for manifest/blob resolution.
	OllamaModels string `json:"ollama_models"`

	// Model name as clients send it -> profile name. Names not listed go to ollama.
	Models   map[string]string  `json:"models"`
	Profiles map[string]Profile `json:"profiles"`

	Cvoice Cvoice `json:"cvoice"`
	Call   Call   `json:"call"`
}

type Cvoice struct {
	URL       string `json:"url"`
	TokenFile string `json:"token_file"` // cvoice's config.toml; [server] token
	Unit      string `json:"unit"`       // user unit, restarted as a fallback when /yield is absent
}

type Call struct {
	ReserveMiB int      `json:"reserve_mib"`
	LeaseTTLS  int      `json:"lease_ttl_s"`
	HealthURL  string   `json:"health_url"` // hotline-calld health: backstop for a missing begin/end
	Models     []string `json:"models"`     // ollama models the call itself uses; never evicted during a call
	// EvictInflight: during a call, stop an LLM even mid-generation. A call is
	// interactive and an LLM client can retry; set false to wait instead.
	EvictInflight bool `json:"evict_inflight"`
}

func Default() Config {
	home, _ := os.UserHomeDir()
	return Config{
		Listen:       "0.0.0.0:11434",
		Ollama:       "http://127.0.0.1:11435",
		MarginMiB:    300,
		StateDir:     filepath.Join(home, ".local/state/gpu-share"),
		LoadWaitS:    20,
		HoldS:        120,
		ReconcileS:   5,
		LlamaServer:  filepath.Join(home, "data/llama-turbo3/build/bin/llama-server"),
		TurboPort:    11436,
		KeepAliveS:   300,
		OllamaModels: "/mnt/windows/Users/Korisnik/.ollama/models",
		Models: map[string]string{
			"piccolo-gorgone:9b":    "piccolo-turbo3-262k",
			"piccolo-gorgone:9b-q8": "piccolo-q8-128k",
		},
		Profiles: map[string]Profile{
			// Measured 2026-11-02 on the RTX 4060: 7384 MiB after load, 7746 MiB peak with a
			// 90k-token prompt (compute buffers grow). KLD vs f16 KV 0.0246.
			"piccolo-turbo3-262k": {OllamaModel: "piccolo-gorgone:9b", Ctx: 262144, CacheK: "turbo3", CacheV: "turbo3", MiB: 7750},
			// 7656 MiB used, KLD 0.0025.
			"piccolo-q8-128k": {OllamaModel: "piccolo-gorgone:9b", Ctx: 131072, CacheK: "q8_0", CacheV: "q8_0", MiB: 7660},
		},
		Cvoice: Cvoice{
			URL:       "http://100.72.2.62:8760",
			TokenFile: filepath.Join(home, ".config/cvoice/config.toml"),
			Unit:      "cvoiced.service",
		},
		Call: Call{
			// Whisper large-v3 int8 ~1.9 GB + cvoiced ~2.4 GB + bge-m3 0.66 GB, per manager-6's measurements.
			ReserveMiB:    5100,
			LeaseTTLS:     90,
			HealthURL:     "http://127.0.0.1:8792/health",
			Models:        []string{"bge-m3:latest", "bge-m3"},
			EvictInflight: true,
		},
	}
}

// Load reads a JSON config over the defaults, so a config file only needs the
// fields it changes.
func Load(path string) (Config, error) {
	c := Default()
	if path == "" {
		return c, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return c, nil
		}
		return c, err
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("%s: %w", path, err)
	}
	for name, prof := range c.Models {
		if _, ok := c.Profiles[prof]; !ok {
			return c, fmt.Errorf("model %q maps to unknown profile %q", name, prof)
		}
	}
	return c, nil
}

func (c Config) IsCallModel(name string) bool {
	for _, m := range c.Call.Models {
		if m == name || strings.TrimSuffix(name, ":latest") == m {
			return true
		}
	}
	return false
}
