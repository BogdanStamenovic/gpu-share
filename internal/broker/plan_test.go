package broker

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func names(p Plan) []string {
	var s []string
	for _, a := range p.Actions {
		s = append(s, key(a.Evict))
	}
	return s
}

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The situation the operator found: cvoiced resident, piccolo wants the card.
func TestCvoicedYieldsToLLM(t *testing.T) {
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 2408, MarginMiB: 300,
		Tenants: []Tenant{{Kind: "cvoiced", Name: "cvoiced", MiB: 2408, CallUse: true}},
		NeedMiB: 7400, TargetKey: "turbo:piccolo"})
	if !eq(names(p), []string{"cvoiced:cvoiced"}) || p.Short || p.Hold {
		t.Fatalf("got %+v", p)
	}
}

func TestCallHoldsBigLoad(t *testing.T) {
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 3100, MarginMiB: 300, CallActive: true, ReserveMiB: 5100,
		Tenants: []Tenant{{Kind: "cvoiced", Name: "cvoiced", MiB: 2408, CallUse: true},
			{Kind: "ollama", Name: "bge-m3:latest", MiB: 664, CallUse: true}},
		NeedMiB: 7400, TargetKey: "turbo:piccolo"})
	if !p.Hold || len(p.Actions) != 0 {
		t.Fatalf("want hold, got %+v", p)
	}
}

func TestCallAllowsSmallLoadButKeepsReserve(t *testing.T) {
	// 8188 - 300 margin - 5100 reserve = 2788 for LLMs. A 1000 MiB model fits.
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 3072, MarginMiB: 300, CallActive: true, ReserveMiB: 5100,
		Tenants: []Tenant{{Kind: "cvoiced", Name: "cvoiced", MiB: 2408, CallUse: true},
			{Kind: "ollama", Name: "bge-m3:latest", MiB: 664, CallUse: true}},
		NeedMiB: 1000, TargetKey: "ollama:qwen2.5:1.5b"})
	if p.Hold || p.Short || len(p.Actions) != 0 {
		t.Fatalf("got %+v", p)
	}
}

func TestCallBeginEvictsLLMEvenInflight(t *testing.T) {
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 7390, MarginMiB: 300, CallActive: true, ReserveMiB: 5100,
		Tenants: []Tenant{{Kind: "turbo", Name: "piccolo", MiB: 7390, Inflight: 1}},
		ForCall: true, EvictInflight: true})
	if !eq(names(p), []string{"turbo:piccolo"}) {
		t.Fatalf("got %+v", p)
	}
	p = MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 7390, MarginMiB: 300, CallActive: true, ReserveMiB: 5100,
		Tenants: []Tenant{{Kind: "turbo", Name: "piccolo", MiB: 7390, Inflight: 1}},
		ForCall: true, EvictInflight: false})
	if len(p.Actions) != 0 || !p.Short {
		t.Fatalf("evict_inflight=false must not evict, got %+v", p)
	}
}

func TestCallBeginWithNothingToDo(t *testing.T) {
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 2408, MarginMiB: 300, CallActive: true, ReserveMiB: 5100,
		Tenants: []Tenant{{Kind: "cvoiced", Name: "cvoiced", MiB: 2408, CallUse: true}},
		ForCall: true})
	if len(p.Actions) != 0 || p.Short {
		t.Fatalf("got %+v", p)
	}
}

func TestLLMLoadNeverInterruptsInflightLLM(t *testing.T) {
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 6454, MarginMiB: 300,
		Tenants: []Tenant{{Kind: "ollama", Name: "qwen3:8b", MiB: 6454, Inflight: 2}},
		NeedMiB: 7400, TargetKey: "turbo:piccolo"})
	if len(p.Actions) != 0 || !p.Short {
		t.Fatalf("got %+v", p)
	}
}

func TestOrderIdleLargestFirstCvoicedBeforeAll(t *testing.T) {
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 8000, MarginMiB: 300,
		Tenants: []Tenant{{Kind: "ollama", Name: "small", MiB: 700}, {Kind: "ollama", Name: "big", MiB: 4000},
			{Kind: "cvoiced", Name: "cvoiced", MiB: 2400, CallUse: true}},
		NeedMiB: 6000, TargetKey: "turbo:piccolo"})
	if !eq(names(p), []string{"cvoiced:cvoiced", "ollama:big"}) {
		t.Fatalf("got %v", names(p))
	}
}

func TestTargetIsNeverEvicted(t *testing.T) {
	p := MakePlan(PlanIn{TotalMiB: 8188, UsedMiB: 8000, MarginMiB: 300,
		Tenants: []Tenant{{Kind: "turbo", Name: "piccolo", MiB: 7390}},
		NeedMiB: 7400, TargetKey: "turbo:piccolo"})
	if len(p.Actions) != 0 {
		t.Fatalf("got %v", names(p))
	}
}

func TestServerLogAppendsAcrossStarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "llama-server-x.log")
	for _, line := range []string{"first run: CUDA error: out of memory\n", "second run\n"} {
		f, err := openServerLog(path)
		if err != nil {
			t.Fatal(err)
		}
		f.WriteString(line)
		f.Close()
	}
	b, _ := os.ReadFile(path)
	if !strings.Contains(string(b), "CUDA error: out of memory") || !strings.Contains(string(b), "second run") {
		t.Fatalf("a restart erased the previous run's log:\n%s", b)
	}
}
