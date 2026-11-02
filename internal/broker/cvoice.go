package broker

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type CvoiceClient struct {
	Cfg  Cvoice
	HTTP *http.Client
}

var tokenRe = regexp.MustCompile(`(?m)^\s*token\s*=\s*"([^"]+)"`)

func (c *CvoiceClient) token() string {
	b, err := os.ReadFile(c.Cfg.TokenFile)
	if err != nil {
		return ""
	}
	if m := tokenRe.FindSubmatch(b); m != nil {
		return string(m[1])
	}
	return ""
}

// MainPID of the cvoiced unit, 0 if it is not running.
func (c *CvoiceClient) MainPID(ctx context.Context) int {
	out, err := exec.CommandContext(ctx, "systemctl", "--user", "show", "-p", "MainPID", "--value", c.Cfg.Unit).Output()
	if err != nil {
		return 0
	}
	pid, _ := strconv.Atoi(strings.TrimSpace(string(out)))
	return pid
}

// Yield makes cvoiced give its VRAM back. It uses POST /yield, which exits the
// process after in-flight synthesis; systemd restarts it empty. /unload is
// not used because it leaks ~780 MB per cycle (cvoice's own measurement), so
// only a process exit is a full reclaim. A cvoiced without /yield is
// restarted through systemd instead, which is the same thing done less politely.
func (c *CvoiceClient) Yield(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", c.Cfg.URL+"/yield?wait=10", nil)
	if t := c.token(); t != "" {
		req.Header.Set("Authorization", "Bearer "+t)
	}
	resp, err := c.HTTP.Do(req)
	if err == nil {
		resp.Body.Close()
		switch {
		case resp.StatusCode == 200:
			return "yield", nil
		case resp.StatusCode == 409:
			return "", fmt.Errorf("cvoiced busy (409), not yielding")
		}
	}
	if err := exec.CommandContext(ctx, "systemctl", "--user", "restart", c.Cfg.Unit).Run(); err != nil {
		return "", fmt.Errorf("restart %s: %w", c.Cfg.Unit, err)
	}
	return "restart", nil
}
