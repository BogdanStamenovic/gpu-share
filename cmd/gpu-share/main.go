// gpu-share: one front door for the GPU. It sits on ollama's port, decides
// who gets the 8 GB card, and routes chosen models to the turbo3 llama.cpp fork.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/BogdanStamenovic/gpu-share/internal/broker"
)

const usage = `gpu-share - GPU admission front door for ollama, the turbo3 llama-server and cvoiced

usage:
  gpu-share serve                 run the broker (normally via the gpu-share user unit)
  gpu-share status                what holds VRAM, live calls, recent evictions
  gpu-share plan MODEL            dry run: what loading MODEL would evict, without doing it
  gpu-share args MODEL            the llama-server command line a routed MODEL would get
  gpu-share call begin [ID]       reserve the call budget (prints the lease)
  gpu-share call renew LEASE
  gpu-share call end LEASE
  gpu-share config                print the effective config as JSON

flags (before the command):
  -config PATH   default ~/.config/gpu-share/config.json (missing file = built-in defaults)
  -url URL       broker to talk to for client commands (default from config listen)

exit codes: 0 ok, 1 runtime failure, 2 usage error
`

func main() {
	home, _ := os.UserHomeDir()
	cfgPath := flag.String("config", filepath.Join(home, ".config/gpu-share/config.json"), "")
	urlFlag := flag.String("url", "", "")
	flag.Usage = func() { fmt.Fprint(os.Stderr, usage) }
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	cfg, err := broker.Load(*cfgPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "gpu-share:", err)
		os.Exit(2)
	}
	base := *urlFlag
	if base == "" {
		hp := cfg.Listen
		if strings.HasPrefix(hp, "0.0.0.0:") {
			hp = "127.0.0.1:" + strings.TrimPrefix(hp, "0.0.0.0:")
		}
		base = "http://" + hp
	}

	switch args[0] {
	case "serve":
		os.Exit(serve(cfg))
	case "status":
		os.Exit(client("GET", base+"/gpu-share/status", nil))
	case "plan":
		if len(args) != 2 {
			die2("plan needs MODEL")
		}
		os.Exit(client("GET", base+"/gpu-share/plan?model="+args[1], nil))
	case "args":
		if len(args) != 2 {
			die2("args needs MODEL")
		}
		prof, ok := cfg.Models[args[1]]
		if !ok {
			fmt.Fprintf(os.Stderr, "%s is not routed to llama-server; it goes to ollama\n", args[1])
			os.Exit(1)
		}
		b := broker.New(cfg)
		a, err := b.Turbo.Args(prof, cfg.Profiles[prof], []string{args[1]})
		if err != nil {
			fmt.Fprintln(os.Stderr, "gpu-share:", err)
			os.Exit(1)
		}
		fmt.Println(cfg.LlamaServer + " " + strings.Join(a, " "))
	case "call":
		if len(args) < 2 {
			die2("call begin|renew|end")
		}
		v := map[string]string{}
		switch args[1] {
		case "begin":
			if len(args) > 2 {
				v["id"] = args[2]
			}
		case "renew", "end":
			if len(args) != 3 {
				die2("call " + args[1] + " needs LEASE")
			}
			v["lease"] = args[2]
		default:
			die2("call begin|renew|end")
		}
		body, _ := json.Marshal(v)
		os.Exit(client("POST", base+"/gpu-share/call/"+args[1], body))
	case "config":
		out, _ := json.MarshalIndent(cfg, "", "  ")
		fmt.Println(string(out))
	default:
		die2("unknown command " + args[0])
	}
}

func die2(msg string) {
	fmt.Fprintln(os.Stderr, "gpu-share:", msg)
	fmt.Fprint(os.Stderr, usage)
	os.Exit(2)
}

func client(method, url string, body []byte) int {
	req, _ := http.NewRequest(method, url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		fmt.Fprintln(os.Stderr, "gpu-share: broker not reachable:", err)
		return 1
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var pretty bytes.Buffer
	if json.Indent(&pretty, b, "", "  ") == nil {
		b = pretty.Bytes()
	}
	fmt.Println(string(b))
	if resp.StatusCode/100 != 2 {
		return 1
	}
	return 0
}

func serve(cfg broker.Config) int {
	b := broker.New(cfg)
	s, err := broker.NewServer(b)
	if err != nil {
		log.Print(err)
		return 1
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	go b.Reconcile(ctx)
	srv := &http.Server{Addr: cfg.Listen, Handler: s}
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		srv.Shutdown(sctx)
	}()
	log.Printf("gpu-share: listening on %s, ollama at %s, llama-server port %d", cfg.Listen, cfg.Ollama, cfg.TurboPort)
	err = srv.ListenAndServe()
	b.Turbo.Stop()
	if err != nil && err != http.ErrServerClosed {
		log.Print(err)
		return 1
	}
	return 0
}
