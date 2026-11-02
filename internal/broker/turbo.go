package broker

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// Turbo owns at most one llama-server child (one profile at a time: the card
// holds one 9B model with a long context, never two).
type Turbo struct {
	Bin      string
	Port     int
	StateDir string
	Ollama   *Ollama

	mu      sync.Mutex
	cmd     *exec.Cmd
	profile string
	done    chan struct{}
	started time.Time
}

func (t *Turbo) URL() string { return fmt.Sprintf("http://127.0.0.1:%d", t.Port) }

// Running returns the loaded profile name, or "".
func (t *Turbo) Running() (string, int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.cmd == nil {
		return "", 0
	}
	return t.profile, t.cmd.Process.Pid
}

var samplingFlags = map[string]string{
	"temperature": "--temp", "top_k": "--top-k", "top_p": "--top-p", "min_p": "--min-p",
	"presence_penalty": "--presence-penalty", "frequency_penalty": "--frequency-penalty",
	"repeat_penalty": "--repeat-penalty", "repeat_last_n": "--repeat-last-n",
}

func (t *Turbo) Args(name string, p Profile, aliases []string) ([]string, error) {
	gguf := p.GGUF
	if gguf == "" {
		var err error
		if gguf, err = t.Ollama.ResolveGGUF(p.OllamaModel); err != nil {
			return nil, err
		}
	}
	args := []string{"-m", gguf, "-c", strconv.Itoa(p.Ctx), "-ngl", "99", "-fa", "on",
		"-ctk", p.CacheK, "-ctv", p.CacheV, "-np", "1",
		"--host", "127.0.0.1", "--port", strconv.Itoa(t.Port), "--jinja"}
	for _, a := range aliases {
		args = append(args, "--alias", a)
	}
	if p.OllamaModel != "" {
		for k, v := range t.Ollama.Params(p.OllamaModel) {
			if f, ok := samplingFlags[k]; ok {
				args = append(args, f, fmt.Sprint(v))
			}
		}
	}
	return append(args, p.ExtraArgs...), nil
}

// serverLogMax is when a profile's log is rotated to .1 (one generation kept).
const serverLogMax = 32 << 20

// openServerLog appends rather than truncates. os.Create here erased each
// crash's own CUDA error the moment the broker restarted the server
// (2026-11-03: six aborts, no error text left), so a restart now adds a header
// and keeps what came before; one rotated generation bounds the size.
func openServerLog(path string) (*os.File, error) {
	if st, err := os.Stat(path); err == nil && st.Size() > serverLogMax {
		_ = os.Rename(path, path+".1")
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(f, "\n===== gpu-share: llama-server start %s =====\n", time.Now().Format(time.RFC3339))
	return f, nil
}

// Start launches the profile and waits until /health answers 200.
func (t *Turbo) Start(ctx context.Context, name string, p Profile, aliases []string) error {
	args, err := t.Args(name, p, aliases)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(t.StateDir, 0o755); err != nil {
		return err
	}
	logf, err := openServerLog(filepath.Join(t.StateDir, "llama-server-"+name+".log"))
	if err != nil {
		return err
	}
	cmd := exec.Command(t.Bin, args...)
	cmd.Stdout, cmd.Stderr = logf, logf
	// If the broker dies, the child must not keep 7 GB of VRAM with nobody to stop it.
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGTERM}
	if err := cmd.Start(); err != nil {
		logf.Close()
		return err
	}
	done := make(chan struct{})
	t.mu.Lock()
	t.cmd, t.profile, t.done, t.started = cmd, name, done, time.Now()
	t.mu.Unlock()
	go func() { cmd.Wait(); logf.Close(); t.clear(cmd); close(done) }()
	log.Printf("turbo: started %s pid %d (ctx %d, K %s, V %s)", name, cmd.Process.Pid, p.Ctx, p.CacheK, p.CacheV)

	deadline := time.Now().Add(240 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case <-done:
			t.clear(cmd)
			return fmt.Errorf("llama-server %s exited during load; see %s", name, logf.Name())
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
		resp, err := http.Get(t.URL() + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				log.Printf("turbo: %s ready in %.1fs", name, time.Since(t.started).Seconds())
				return nil
			}
		}
	}
	t.Stop()
	return fmt.Errorf("llama-server %s not ready after 240s", name)
}

func (t *Turbo) clear(cmd *exec.Cmd) {
	t.mu.Lock()
	if t.cmd == cmd {
		t.cmd, t.profile = nil, ""
	}
	t.mu.Unlock()
}

// Stop terminates the child and waits until it has exited (and so released VRAM).
func (t *Turbo) Stop() {
	t.mu.Lock()
	cmd, done, name := t.cmd, t.done, t.profile
	t.mu.Unlock()
	if cmd == nil {
		return
	}
	cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		cmd.Process.Kill()
		<-done
	}
	t.clear(cmd)
	log.Printf("turbo: stopped %s", name)
}
