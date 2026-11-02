package broker

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sort"
	"sync"
	"time"
)

var ErrHeld = errors.New("held: a live call needs the GPU")

type Broker struct {
	Cfg    Config
	Ollama *Ollama
	Turbo  *Turbo
	Cvoice *CvoiceClient

	loadMu sync.Mutex // one eviction+load at a time

	mu        sync.Mutex
	inflight  map[string]int
	lastUsed  map[string]time.Time
	learned   map[string]int // ollama model -> MiB seen fully loaded
	leases    map[string]time.Time
	healthIn  bool // hotline-calld says a call is up
	callWake  chan struct{}
	lastEvict []string
}

func New(cfg Config) *Broker {
	hc := &http.Client{Timeout: 30 * time.Second}
	o := &Ollama{Base: cfg.Ollama, ModelsDir: cfg.OllamaModels, HTTP: hc}
	return &Broker{
		Cfg:      cfg,
		Ollama:   o,
		Turbo:    &Turbo{Bin: cfg.LlamaServer, Port: cfg.TurboPort, StateDir: cfg.StateDir, Ollama: o},
		Cvoice:   &CvoiceClient{Cfg: cfg.Cvoice, HTTP: hc},
		inflight: map[string]int{}, lastUsed: map[string]time.Time{},
		learned: map[string]int{}, leases: map[string]time.Time{},
		callWake: make(chan struct{}),
	}
}

func (b *Broker) callActiveLocked() bool {
	now := time.Now()
	for id, exp := range b.leases {
		if now.After(exp) {
			log.Printf("call: lease %s expired without end", id)
			delete(b.leases, id)
		}
	}
	return len(b.leases) > 0 || b.healthIn
}

func (b *Broker) CallActive() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.callActiveLocked()
}

// wakeCallWaiters is called with b.mu held whenever call state may have changed.
func (b *Broker) wakeCallWaitersLocked() {
	close(b.callWake)
	b.callWake = make(chan struct{})
}

// Snapshot gathers what holds VRAM right now.
func (b *Broker) Snapshot(ctx context.Context) (GPU, []Tenant, error) {
	g, err := QueryGPU(ctx)
	if err != nil {
		return g, nil, err
	}
	cvPID := b.Cvoice.MainPID(ctx)
	ps, psErr := b.Ollama.PS(ctx)
	prof, pid := b.Turbo.Running()
	var ts []Tenant
	b.mu.Lock()
	defer b.mu.Unlock()
	if cvPID > 0 && g.Procs[cvPID] > 0 {
		ts = append(ts, Tenant{Kind: "cvoiced", Name: "cvoiced", MiB: g.Procs[cvPID], CallUse: true})
	}
	if prof != "" {
		ts = append(ts, Tenant{Kind: "turbo", Name: prof, MiB: g.Procs[pid], Inflight: b.inflight["turbo:"+prof]})
	}
	if psErr == nil {
		for _, m := range ps {
			if m.Size-m.SizeVRAM < splitSlack {
				b.learned[m.Name] = int(m.Size >> 20)
			}
			ts = append(ts, Tenant{Kind: "ollama", Name: m.Name, MiB: int(m.SizeVRAM >> 20),
				CallUse: b.Cfg.IsCallModel(m.Name), Inflight: b.inflight["ollama:"+m.Name]})
		}
	}
	return g, ts, nil
}

// splitSlack: piccolo keeps ~546 MiB of token embeddings CPU-mapped even when
// all 33 layers are on the GPU, so size - size_vram is never 0. A real split
// (18/33 layers) leaves ~3 GB on the CPU.
const splitSlack = 800 << 20

// Target is what a request will load.
type Target struct {
	Kind    string // "turbo" or "ollama"
	Name    string // profile name or ollama model
	Aliases []string
}

func (t Target) Key() string { return t.Kind + ":" + t.Name }

func (b *Broker) need(t Target) int {
	if t.Kind == "turbo" {
		return b.Cfg.Profiles[t.Name].MiB
	}
	b.mu.Lock()
	n := b.learned[t.Name]
	b.mu.Unlock()
	if n > 0 {
		return n
	}
	// Never seen loaded: weights plus a context allowance.
	return b.Ollama.WeightsMiB(t.Name)*115/100 + 512
}

func (b *Broker) resident(t Target, ts []Tenant) bool {
	for _, x := range ts {
		if key(x) == t.Key() {
			return true
		}
	}
	return false
}

// PlanFor shows what Admit would do, without doing it.
func (b *Broker) PlanFor(ctx context.Context, t Target) (Plan, []Tenant, error) {
	g, ts, err := b.Snapshot(ctx)
	if err != nil {
		return Plan{}, ts, err
	}
	if b.resident(t, ts) {
		return Plan{Reason: "already resident"}, ts, nil
	}
	in := PlanIn{TotalMiB: g.TotalMiB, UsedMiB: g.UsedMiB, MarginMiB: b.Cfg.MarginMiB, Tenants: ts,
		CallActive: b.CallActive(), ReserveMiB: b.Cfg.Call.ReserveMiB, NeedMiB: b.need(t), TargetKey: t.Key()}
	p := MakePlan(in)
	// Only one llama-server runs at a time, so another profile must go even if
	// the arithmetic says otherwise.
	if t.Kind == "turbo" && !p.Hold {
		for _, x := range ts {
			if x.Kind == "turbo" && x.Name != t.Name {
				found := false
				for _, a := range p.Actions {
					found = found || key(a.Evict) == key(x)
				}
				if !found {
					p.Actions = append([]Action{{Evict: x, Reason: "one llama-server profile at a time"}}, p.Actions...)
				}
			}
		}
	}
	return p, ts, nil
}

// Admit makes room for t and, for a turbo target, starts it. It returns once
// the request may be forwarded. ErrHeld means a call is up and the load
// cannot fit beside it within HoldS.
func (b *Broker) Admit(ctx context.Context, t Target) error {
	hold := time.Now().Add(time.Duration(b.Cfg.HoldS) * time.Second)
	for {
		b.loadMu.Lock()
		p, ts, err := b.PlanFor(ctx, t)
		if err != nil {
			b.loadMu.Unlock()
			log.Printf("admit %s: snapshot failed (%v); forwarding unmanaged", t.Key(), err)
			return nil // degrade: the GPU state is unknown, but the request still works
		}
		if p.Hold {
			b.mu.Lock()
			wake := b.callWake
			b.mu.Unlock()
			b.loadMu.Unlock()
			log.Printf("admit %s: %s; holding", t.Key(), p.Reason)
			select {
			case <-wake:
				continue
			case <-time.After(time.Until(hold)):
				return ErrHeld
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if busy := b.otherTurboBusy(t, ts); busy {
			b.loadMu.Unlock()
			if time.Now().After(hold) {
				return ErrHeld
			}
			select {
			case <-time.After(time.Second):
				continue
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		err = b.execute(ctx, p, ts, t)
		b.loadMu.Unlock()
		return err
	}
}

func (b *Broker) otherTurboBusy(t Target, ts []Tenant) bool {
	if t.Kind != "turbo" {
		return false
	}
	for _, x := range ts {
		if x.Kind == "turbo" && x.Name != t.Name && x.Inflight > 0 {
			return true
		}
	}
	return false
}

func (b *Broker) execute(ctx context.Context, p Plan, ts []Tenant, t Target) error {
	if p.Reason == "already resident" {
		return nil
	}
	b.evict(ctx, p.Actions)
	if p.Short {
		log.Printf("admit %s: %s; loading anyway (it may split onto the CPU)", t.Key(), p.Reason)
	}
	if t.Kind == "turbo" {
		b.waitFree(ctx, p.NeedMiB)
		return b.Turbo.Start(ctx, t.Name, b.Cfg.Profiles[t.Name], t.Aliases)
	}
	b.waitFree(ctx, p.NeedMiB)
	return nil
}

// waitFree polls until the evicted tenants have actually released memory:
// cvoiced restarts, and ollama unloads asynchronously.
func (b *Broker) waitFree(ctx context.Context, need int) {
	deadline := time.Now().Add(time.Duration(b.Cfg.LoadWaitS) * time.Second)
	for time.Now().Before(deadline) {
		g, err := QueryGPU(ctx)
		if err != nil || g.FreeMiB()-b.Cfg.MarginMiB >= need {
			return
		}
		time.Sleep(250 * time.Millisecond)
	}
	log.Printf("waitFree: %d MiB not free after %ds; proceeding", need, b.Cfg.LoadWaitS)
}

func (b *Broker) evict(ctx context.Context, acts []Action) {
	for _, a := range acts {
		t := a.Evict
		var err error
		how := ""
		switch t.Kind {
		case "cvoiced":
			how, err = b.Cvoice.Yield(ctx)
		case "turbo":
			b.Turbo.Stop()
			how = "stop"
		case "ollama":
			err = b.Ollama.Unload(ctx, t.Name)
			how = "keep_alive=0"
		}
		msg := fmt.Sprintf("%s %s:%s (%d MiB) via %s: %s", time.Now().Format("15:04:05"), t.Kind, t.Name, t.MiB, how, a.Reason)
		if err != nil {
			msg += " FAILED: " + err.Error()
		}
		log.Print("evict: " + msg)
		b.mu.Lock()
		b.lastEvict = append(b.lastEvict, msg)
		if len(b.lastEvict) > 20 {
			b.lastEvict = b.lastEvict[len(b.lastEvict)-20:]
		}
		b.mu.Unlock()
	}
}

func (b *Broker) Begin(k string) func() {
	b.mu.Lock()
	b.inflight[k]++
	b.mu.Unlock()
	return func() {
		b.mu.Lock()
		b.inflight[k]--
		b.lastUsed[k] = time.Now()
		b.mu.Unlock()
	}
}

// --- calls ---

type Lease struct {
	ID       string `json:"lease"`
	TTLS     int    `json:"ttl_s"`
	Evicting bool   `json:"evicting"`
	Plan     *Plan  `json:"plan,omitempty"`
}

// CallBegin registers a call and frees its reserve in the background. It
// returns at once: the pickup must never wait on an eviction.
func (b *Broker) CallBegin(id string) Lease {
	if id == "" {
		var r [6]byte
		rand.Read(r[:])
		id = hex.EncodeToString(r[:])
	}
	ttl := time.Duration(b.Cfg.Call.LeaseTTLS) * time.Second
	b.mu.Lock()
	b.leases[id] = time.Now().Add(ttl)
	b.wakeCallWaitersLocked()
	b.mu.Unlock()
	log.Printf("call: begin %s", id)
	l := Lease{ID: id, TTLS: b.Cfg.Call.LeaseTTLS}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if p, err := b.callPlan(ctx); err == nil {
		l.Plan = &p
		if len(p.Actions) > 0 {
			l.Evicting = true
			go b.makeRoomForCall()
		}
	}
	return l
}

func (b *Broker) callPlan(ctx context.Context) (Plan, error) {
	g, ts, err := b.Snapshot(ctx)
	if err != nil {
		return Plan{}, err
	}
	return MakePlan(PlanIn{TotalMiB: g.TotalMiB, UsedMiB: g.UsedMiB, MarginMiB: b.Cfg.MarginMiB, Tenants: ts,
		CallActive: true, ReserveMiB: b.Cfg.Call.ReserveMiB, ForCall: true, EvictInflight: b.Cfg.Call.EvictInflight}), nil
}

func (b *Broker) makeRoomForCall() {
	b.loadMu.Lock()
	defer b.loadMu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if p, err := b.callPlan(ctx); err == nil {
		b.evict(ctx, p.Actions)
	}
}

func (b *Broker) CallRenew(id string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.leases[id]; !ok {
		return false
	}
	b.leases[id] = time.Now().Add(time.Duration(b.Cfg.Call.LeaseTTLS) * time.Second)
	return true
}

func (b *Broker) CallEnd(id string) {
	b.mu.Lock()
	delete(b.leases, id)
	b.wakeCallWaitersLocked()
	b.mu.Unlock()
	log.Printf("call: end %s", id)
}

// --- the feedback loop ---

// Reconcile runs forever. It is the part that notices the world disagrees
// with what the broker believes, and repairs it:
//   - a call that skipped begin, or a calld that crashed mid-call (health poll)
//   - an ollama model left split across CPU and GPU, which stays slow until
//     it unloads: evict it once there is room for a full reload
//   - a turbo llama-server idle past keep_alive
func (b *Broker) Reconcile(ctx context.Context) {
	tick := time.NewTicker(time.Duration(b.Cfg.ReconcileS) * time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		b.pollCallHealth(ctx)
		b.stopIdleTurbo()
		b.fixSplit(ctx)
	}
}

func (b *Broker) pollCallHealth(ctx context.Context) {
	if b.Cfg.Call.HealthURL == "" {
		return
	}
	c, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(c, "GET", b.Cfg.Call.HealthURL, nil)
	in := false
	if resp, err := http.DefaultClient.Do(req); err == nil {
		var h struct {
			Active  int `json:"active_calls"`
			Inbound struct {
				InCall bool `json:"inCall"`
			} `json:"inbound"`
		}
		json.NewDecoder(resp.Body).Decode(&h)
		resp.Body.Close()
		in = h.Active > 0 || h.Inbound.InCall
	}
	b.mu.Lock()
	changed := in != b.healthIn
	hadLease := len(b.leases) > 0
	b.healthIn = in
	if changed {
		b.wakeCallWaitersLocked()
	}
	b.mu.Unlock()
	if changed {
		log.Printf("call: health says in_call=%v (lease held: %v)", in, hadLease)
		if in && !hadLease {
			go b.makeRoomForCall()
		}
	}
}

func (b *Broker) stopIdleTurbo() {
	prof, _ := b.Turbo.Running()
	if prof == "" {
		return
	}
	k := "turbo:" + prof
	b.mu.Lock()
	idle := b.inflight[k] == 0 && time.Since(b.lastUsed[k]) > time.Duration(b.Cfg.KeepAliveS)*time.Second
	if b.lastUsed[k].IsZero() {
		b.lastUsed[k] = time.Now() // started but never used: count from now
		idle = false
	}
	b.mu.Unlock()
	if idle && b.loadMu.TryLock() {
		log.Printf("turbo: %s idle %ds, stopping", prof, b.Cfg.KeepAliveS)
		b.Turbo.Stop()
		b.loadMu.Unlock()
	}
}

func (b *Broker) fixSplit(ctx context.Context) {
	if b.CallActive() {
		return
	}
	ps, err := b.Ollama.PS(ctx)
	if err != nil {
		return
	}
	g, err := QueryGPU(ctx)
	if err != nil {
		return
	}
	for _, m := range ps {
		if m.Size-m.SizeVRAM < splitSlack {
			continue
		}
		b.mu.Lock()
		busy := b.inflight["ollama:"+m.Name] > 0
		b.mu.Unlock()
		// Would it fit fully now, counting what it already holds?
		fits := g.FreeMiB()-b.Cfg.MarginMiB+int(m.SizeVRAM>>20) >= int((m.Size-splitSlack)>>20)
		if busy || !fits {
			continue
		}
		if !b.loadMu.TryLock() {
			return
		}
		b.evict(ctx, []Action{{Evict: Tenant{Kind: "ollama", Name: m.Name, MiB: int(m.SizeVRAM >> 20)},
			Reason: fmt.Sprintf("resident split (%d of %d MiB on GPU) and now fits fully; next request reloads it on the GPU",
				m.SizeVRAM>>20, m.Size>>20)}})
		b.loadMu.Unlock()
		return
	}
}

// Status is what `gpu-share status` prints.
func (b *Broker) Status(ctx context.Context) map[string]any {
	g, ts, err := b.Snapshot(ctx)
	sort.Slice(ts, func(i, j int) bool { return ts[i].MiB > ts[j].MiB })
	b.mu.Lock()
	defer b.mu.Unlock()
	leases := map[string]string{}
	for id, exp := range b.leases {
		leases[id] = time.Until(exp).Round(time.Second).String()
	}
	prof, pid := b.Turbo.Running()
	st := map[string]any{
		"gpu":              map[string]int{"total_mib": g.TotalMiB, "used_mib": g.UsedMiB, "free_mib": g.FreeMiB()},
		"tenants":          ts,
		"call_active":      b.callActiveLocked(),
		"call_leases":      leases,
		"call_health_in":   b.healthIn,
		"turbo":            map[string]any{"profile": prof, "pid": pid},
		"recent_evictions": b.lastEvict,
	}
	if err != nil {
		st["error"] = err.Error()
	}
	return st
}
