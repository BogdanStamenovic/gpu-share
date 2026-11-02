package broker

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type GPU struct {
	TotalMiB int
	UsedMiB  int
	Procs    map[int]int // pid -> MiB
}

func (g GPU) FreeMiB() int { return g.TotalMiB - g.UsedMiB }

// QueryGPU asks nvidia-smi. NVML would avoid the fork, but nvidia-smi is
// what every other tool on this box reads, so the numbers agree with theirs.
func QueryGPU(ctx context.Context) (GPU, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return GPU{}, fmt.Errorf("nvidia-smi: %w", err)
	}
	f := strings.Split(strings.TrimSpace(strings.Split(string(out), "\n")[0]), ",")
	if len(f) != 2 {
		return GPU{}, fmt.Errorf("nvidia-smi: unexpected %q", out)
	}
	g := GPU{Procs: map[int]int{}}
	g.TotalMiB, _ = strconv.Atoi(strings.TrimSpace(f[0]))
	g.UsedMiB, _ = strconv.Atoi(strings.TrimSpace(f[1]))
	out, err = exec.CommandContext(ctx, "nvidia-smi",
		"--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output()
	if err == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			p := strings.Split(line, ",")
			if len(p) != 2 {
				continue
			}
			pid, e1 := strconv.Atoi(strings.TrimSpace(p[0]))
			mib, e2 := strconv.Atoi(strings.TrimSpace(p[1]))
			if e1 == nil && e2 == nil {
				g.Procs[pid] += mib
			}
		}
	}
	return g, nil
}
