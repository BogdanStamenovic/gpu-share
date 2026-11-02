package broker

import (
	"fmt"
	"sort"
)

// Tenant is anything holding VRAM that the broker can make let go.
type Tenant struct {
	Kind     string `json:"kind"` // "ollama", "turbo", "cvoiced"
	Name     string `json:"name"`
	MiB      int    `json:"mib"`
	CallUse  bool   `json:"call_use"` // part of a live call's budget (cvoiced, bge-m3)
	Inflight int    `json:"inflight"`
}

type Action struct {
	Evict  Tenant `json:"evict"`
	Reason string `json:"reason"`
}

type PlanIn struct {
	TotalMiB, UsedMiB, MarginMiB int
	Tenants                      []Tenant
	CallActive                   bool
	ReserveMiB                   int
	// For a load: what it needs. For a call begin: Need is ignored.
	NeedMiB       int
	TargetKey     string // Kind+":"+Name of the thing being loaded, never evicted
	ForCall       bool   // planning a call's reservation rather than an LLM load
	EvictInflight bool   // only consulted when ForCall
}

type Plan struct {
	Actions []Action `json:"actions"`
	FreeMiB int      `json:"free_mib"` // usable after the actions
	NeedMiB int      `json:"need_mib"`
	// Hold: the load cannot fit while the call is up; wait for it to end.
	Hold bool `json:"hold"`
	// Short: even after every allowed eviction it does not fit (in-flight work
	// is in the way, or it is simply too big). The load proceeds degraded.
	Short  bool   `json:"short"`
	Reason string `json:"reason"`
}

func key(t Tenant) string { return t.Kind + ":" + t.Name }

// MakePlan decides which tenants to evict. Pure, so the policy is testable
// without a GPU, and `gpu-share plan` can show it without doing it.
//
// Policy, in order:
//   - cvoiced yields to any LLM load, but never during a call.
//   - During a call the call's reserve is off limits: an LLM load that cannot fit
//     beside it is held, not split, until hang-up.
//   - An LLM load never interrupts another LLM's in-flight request; a call may.
//   - Idle tenants go before busy ones, larger before smaller.
func MakePlan(in PlanIn) Plan {
	free := in.TotalMiB - in.UsedMiB - in.MarginMiB
	callUsed := 0
	for _, t := range in.Tenants {
		if t.CallUse && in.CallActive {
			callUsed += t.MiB
		}
	}
	// Room the call may still grow into, e.g. Whisper loading at ring time,
	// which no tenant entry accounts for yet.
	callGrowth := 0
	if in.CallActive && in.ReserveMiB > callUsed {
		callGrowth = in.ReserveMiB - callUsed
	}

	need := in.NeedMiB
	if in.ForCall {
		need = callGrowth
		callGrowth = 0
	}
	p := Plan{NeedMiB: need}
	if in.CallActive && !in.ForCall && need > in.TotalMiB-in.MarginMiB-in.ReserveMiB {
		p.Hold = true
		p.Reason = fmt.Sprintf("needs %d MiB; only %d MiB exist beside the call's %d MiB reserve",
			need, in.TotalMiB-in.MarginMiB-in.ReserveMiB, in.ReserveMiB)
		return p
	}
	avail := free - callGrowth

	var cands []Tenant
	for _, t := range in.Tenants {
		if key(t) == in.TargetKey || t.MiB <= 0 {
			continue
		}
		if in.CallActive && t.CallUse {
			continue
		}
		if t.Inflight > 0 && !(in.ForCall && in.EvictInflight) {
			continue
		}
		cands = append(cands, t)
	}
	rank := func(t Tenant) int {
		if t.Kind == "cvoiced" {
			return 0 // cheapest to bring back: ~6 s, and calls warm it at ring time anyway
		}
		if t.Inflight == 0 {
			return 1
		}
		return 2
	}
	sort.SliceStable(cands, func(i, j int) bool {
		ri, rj := rank(cands[i]), rank(cands[j])
		if ri != rj {
			return ri < rj
		}
		return cands[i].MiB > cands[j].MiB
	})
	for _, t := range cands {
		if avail >= need {
			break
		}
		why := "LLM load needs the room"
		if in.ForCall {
			why = "live call needs the room"
		}
		p.Actions = append(p.Actions, Action{Evict: t, Reason: why})
		avail += t.MiB
	}
	p.FreeMiB = avail
	if avail < need {
		p.Short = true
		p.Reason = fmt.Sprintf("short by %d MiB after all allowed evictions", need-avail)
	}
	return p
}
