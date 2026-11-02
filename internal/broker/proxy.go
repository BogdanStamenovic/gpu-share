package broker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type Server struct {
	B        *Broker
	toOllama *httputil.ReverseProxy
	toTurbo  *httputil.ReverseProxy
}

func NewServer(b *Broker) (*Server, error) {
	ou, err := url.Parse(b.Cfg.Ollama)
	if err != nil {
		return nil, err
	}
	tu, _ := url.Parse(b.Turbo.URL())
	mk := func(u *url.URL) *httputil.ReverseProxy {
		p := httputil.NewSingleHostReverseProxy(u)
		p.FlushInterval = -1 // stream tokens as they come
		p.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			if !errors.Is(err, context.Canceled) {
				log.Printf("proxy %s %s -> %s: %v", r.Method, r.URL.Path, u.Host, err)
			}
			jsonErr(w, http.StatusBadGateway, "gpu-share: backend "+u.Host+": "+err.Error())
		}
		return p
	}
	return &Server{B: b, toOllama: mk(ou), toTurbo: mk(tu)}, nil
}

func jsonErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": msg})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// Paths whose requests the turbo llama-server answers itself. ollama-native
// /api/chat and /api/generate for a routed model go to ollama instead (with
// llama-server stopped first): llama-server does not speak that protocol, and
// translating it is not built yet.
var turboPaths = map[string]bool{"/v1/chat/completions": true, "/v1/completions": true}

var modelPaths = map[string]bool{"/api/chat": true, "/api/generate": true, "/api/embed": true,
	"/api/embeddings": true, "/v1/chat/completions": true, "/v1/completions": true, "/v1/embeddings": true}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.HasPrefix(r.URL.Path, "/gpu-share/"):
		s.control(w, r)
		return
	case r.URL.Path == "/api/ps" && r.Method == "GET":
		s.ps(w, r)
		return
	case (r.URL.Path == "/api/tags" || r.URL.Path == "/v1/models") && r.Method == "GET":
		s.listing(w, r)
		return
	case r.URL.Path == "/api/show" && r.Method == "POST":
		s.show(w, r)
		return
	case r.Method == "POST" && modelPaths[r.URL.Path]:
		s.model(w, r)
		return
	}
	s.toOllama.ServeHTTP(w, r)
}

func readBody(r *http.Request) ([]byte, map[string]any, error) {
	b, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		return nil, nil, err
	}
	var m map[string]any
	json.Unmarshal(b, &m)
	return b, m, nil
}

func setBody(r *http.Request, b []byte) {
	r.Body = io.NopCloser(bytes.NewReader(b))
	r.ContentLength = int64(len(b))
	r.Header.Set("Content-Length", strconv.Itoa(len(b)))
}

func (s *Server) aliasesFor(profile string) []string {
	var a []string
	for name, p := range s.B.Cfg.Models {
		if p == profile {
			a = append(a, name)
		}
	}
	return a
}

func isUnload(m map[string]any) bool {
	ka, ok := m["keep_alive"]
	if !ok {
		return false
	}
	switch v := ka.(type) {
	case float64:
		return v == 0
	case string:
		return v == "0" || v == "0s" || v == "0m"
	}
	return false
}

func (s *Server) model(w http.ResponseWriter, r *http.Request) {
	body, m, err := readBody(r)
	if err != nil {
		jsonErr(w, 400, err.Error())
		return
	}
	name, _ := m["model"].(string)
	prof, routed := s.B.Cfg.Models[name]
	var t Target
	var proxy *httputil.ReverseProxy
	switch {
	case routed && turboPaths[r.URL.Path]:
		t = Target{Kind: "turbo", Name: prof, Aliases: s.aliasesFor(prof)}
		proxy = s.toTurbo
	case routed:
		base := s.B.Cfg.Profiles[prof].OllamaModel
		m["model"] = base
		body, _ = json.Marshal(m)
		t = Target{Kind: "ollama", Name: base}
		proxy = s.toOllama
		if p, _ := s.B.Turbo.Running(); p != "" {
			// ollama is about to load the same weights; the two copies cannot share the card.
			s.B.loadMu.Lock()
			s.B.Turbo.Stop()
			s.B.loadMu.Unlock()
		}
	default:
		t = Target{Kind: "ollama", Name: name}
		proxy = s.toOllama
	}
	setBody(r, body)

	if t.Kind == "ollama" && isUnload(m) {
		// An unload request must not trigger a load-side eviction.
		s.toOllama.ServeHTTP(w, r)
		return
	}
	if !s.fastResident(r.Context(), t) {
		err := s.B.Admit(r.Context(), t)
		if err != nil && !errors.Is(err, ErrHeld) && t.Kind == "turbo" {
			// The fork is a quality feature, not a dependency: if llama-server
			// cannot start (binary missing, crash on load), stock ollama still
			// answers, with its own smaller context.
			log.Printf("turbo %s unavailable (%v); falling back to ollama", t.Name, err)
			base := s.B.Cfg.Profiles[t.Name].OllamaModel
			m["model"] = base
			body, _ = json.Marshal(m)
			setBody(r, body)
			t, proxy = Target{Kind: "ollama", Name: base}, s.toOllama
			if !s.fastResident(r.Context(), t) {
				err = s.B.Admit(r.Context(), t)
			} else {
				err = nil
			}
		}
		if err != nil {
			if errors.Is(err, ErrHeld) {
				w.Header().Set("Retry-After", "30")
				jsonErr(w, http.StatusServiceUnavailable, "gpu-share: "+err.Error()+"; retry after the call")
				return
			}
			jsonErr(w, http.StatusServiceUnavailable, "gpu-share: "+err.Error())
			return
		}
	}
	done := s.B.Begin(t.Key())
	defer done()
	proxy.ServeHTTP(w, r)
}

// fastResident skips the nvidia-smi snapshot when the model is already loaded,
// which is the common case and the one where latency matters (bge-m3 embeds
// during a call are ~10 ms; a full snapshot costs more than that).
func (s *Server) fastResident(ctx context.Context, t Target) bool {
	if t.Kind == "turbo" {
		p, _ := s.B.Turbo.Running()
		return p == t.Name
	}
	ps, err := s.B.Ollama.PS(ctx)
	if err != nil {
		return false
	}
	for _, m := range ps {
		if m.Name == t.Name || m.Name == t.Name+":latest" {
			return true
		}
	}
	return false
}

func (s *Server) ps(w http.ResponseWriter, r *http.Request) {
	var out struct {
		Models []map[string]any `json:"models"`
	}
	resp, err := s.B.Ollama.HTTP.Get(s.B.Cfg.Ollama + "/api/ps")
	if err == nil {
		json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
	}
	if prof, pid := s.B.Turbo.Running(); prof != "" {
		g, _ := QueryGPU(r.Context())
		p := s.B.Cfg.Profiles[prof]
		for _, alias := range s.aliasesFor(prof) {
			out.Models = append(out.Models, map[string]any{
				"name": alias, "model": alias, "size": int64(g.Procs[pid]) << 20, "size_vram": int64(g.Procs[pid]) << 20,
				"context_length": p.Ctx, "expires_at": time.Now().Add(time.Duration(s.B.Cfg.KeepAliveS) * time.Second),
				"details": map[string]any{"format": "gguf", "kv_cache": p.CacheK + "/" + p.CacheV, "server": "gpu-share turbo3 llama-server"},
			})
		}
	}
	if out.Models == nil {
		out.Models = []map[string]any{}
	}
	writeJSON(w, out)
}

// listing adds the alias names (e.g. piccolo-gorgone:9b-q8) that only the
// broker knows to ollama's model lists.
func (s *Server) listing(w http.ResponseWriter, r *http.Request) {
	resp, err := s.B.Ollama.HTTP.Get(s.B.Cfg.Ollama + r.URL.Path)
	if err != nil {
		jsonErr(w, 502, err.Error())
		return
	}
	defer resp.Body.Close()
	var m map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		jsonErr(w, 502, err.Error())
		return
	}
	listKey, idKey := "models", "name"
	if r.URL.Path == "/v1/models" {
		listKey, idKey = "data", "id"
	}
	items, _ := m[listKey].([]any)
	have := map[string]map[string]any{}
	for _, it := range items {
		if o, ok := it.(map[string]any); ok {
			if id, _ := o[idKey].(string); id != "" {
				have[id] = o
			}
		}
	}
	for name, prof := range s.B.Cfg.Models {
		if _, ok := have[name]; ok {
			continue
		}
		base, ok := have[s.B.Cfg.Profiles[prof].OllamaModel]
		if !ok {
			continue
		}
		c := map[string]any{}
		for k, v := range base {
			c[k] = v
		}
		c[idKey] = name
		if idKey == "name" {
			c["model"] = name
		}
		items = append(items, c)
	}
	m[listKey] = items
	writeJSON(w, m)
}

var numCtxRe = regexp.MustCompile(`(?m)^(num_ctx\s+)\d+`)

// show answers for a routed model with the context the broker actually serves,
// since clients (OpenHands via litellm) size their prompts from it.
func (s *Server) show(w http.ResponseWriter, r *http.Request) {
	body, m, err := readBody(r)
	if err != nil {
		jsonErr(w, 400, err.Error())
		return
	}
	name, _ := m["model"].(string)
	if name == "" {
		name, _ = m["name"].(string)
	}
	prof, routed := s.B.Cfg.Models[name]
	if !routed {
		setBody(r, body)
		s.toOllama.ServeHTTP(w, r)
		return
	}
	p := s.B.Cfg.Profiles[prof]
	m["model"] = p.OllamaModel
	delete(m, "name")
	body, _ = json.Marshal(m)
	resp, err := s.B.Ollama.HTTP.Post(s.B.Cfg.Ollama+"/api/show", "application/json", bytes.NewReader(body))
	if err != nil {
		jsonErr(w, 502, err.Error())
		return
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || resp.StatusCode != 200 {
		jsonErr(w, resp.StatusCode, "ollama /api/show failed")
		return
	}
	if params, ok := out["parameters"].(string); ok {
		ctx := strconv.Itoa(p.Ctx)
		if numCtxRe.MatchString(params) {
			out["parameters"] = numCtxRe.ReplaceAllString(params, "${1}"+ctx)
		} else {
			out["parameters"] = params + "\nnum_ctx                        " + ctx
		}
	}
	out["gpu_share"] = map[string]any{"profile": prof, "ctx": p.Ctx, "cache_k": p.CacheK, "cache_v": p.CacheV,
		"note": "served by the turbo3 llama.cpp fork on /v1; ollama-native endpoints use stock ollama"}
	writeJSON(w, out)
}

func (s *Server) control(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Lease string `json:"lease"`
		ID    string `json:"id"`
	}
	if r.Method == "POST" {
		json.NewDecoder(r.Body).Decode(&req)
	}
	switch strings.TrimPrefix(r.URL.Path, "/gpu-share/") {
	case "status":
		writeJSON(w, s.B.Status(r.Context()))
	case "plan":
		name := r.URL.Query().Get("model")
		t := Target{Kind: "ollama", Name: name}
		if prof, ok := s.B.Cfg.Models[name]; ok {
			t = Target{Kind: "turbo", Name: prof}
		}
		p, ts, err := s.B.PlanFor(r.Context(), t)
		if err != nil {
			jsonErr(w, 500, err.Error())
			return
		}
		writeJSON(w, map[string]any{"target": t.Key(), "plan": p, "tenants": ts, "call_active": s.B.CallActive()})
	case "call/begin":
		writeJSON(w, s.B.CallBegin(req.ID))
	case "call/renew":
		if !s.B.CallRenew(req.Lease) {
			jsonErr(w, 404, "no such lease (expired?); call begin again")
			return
		}
		writeJSON(w, map[string]any{"ok": true, "ttl_s": s.B.Cfg.Call.LeaseTTLS})
	case "call/end":
		s.B.CallEnd(req.Lease)
		writeJSON(w, map[string]any{"ok": true})
	default:
		jsonErr(w, 404, "unknown gpu-share endpoint")
	}
}
