package a2a

// Test-only fixture compiled inside a clean kagent v0.10.0 checkout. It uses
// the unmodified native executor, ADK function tool, and controller passthrough.
import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"testing"

	old "github.com/a2aproject/a2a-go/a2asrv"
	a2a "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/go-logr/logr"
	native "github.com/kagent-dev/kagent/go/adk/pkg/a2a"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/genai"
)

type progressModel struct {
	final chan struct{}
	calls atomic.Int32
}
type progressWriter struct {
	http.ResponseWriter
	t *testing.T
}

func (w progressWriter) Write(p []byte) (int, error) {
	for _, kind := range []string{"function_call", "function_response", "COMPLETED", "completed"} {
		if bytes.Contains(p, []byte(kind)) {
			w.t.Logf("NATIVE_SSE_EVENT=%s bytes=%d", kind, len(p))
		}
	}
	return w.ResponseWriter.Write(p)
}
func (w progressWriter) Flush()       { w.ResponseWriter.(http.Flusher).Flush() }
func (m *progressModel) Name() string { return "synthetic-progress" }
func (m *progressModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if stream {
			yield(nil, fmt.Errorf("LLM stream must remain false"))
			return
		}
		m.calls.Add(1)
		for _, c := range req.Contents {
			for _, p := range c.Parts {
				if p.FunctionResponse != nil {
					select {
					case <-m.final:
					case <-ctx.Done():
						return
					}
					yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{genai.NewPartFromText("Synthetic final answer")}}}, nil)
					return
				}
			}
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "clock-call", Name: "GetDateTime", Args: map[string]any{}}}}}}, nil)
	}
}

func TestHermesNativeProgress(t *testing.T) {
	if os.Getenv("HERMES_PYTHON") == "" {
		t.Fatal("HERMES_PYTHON required")
	}
	for _, version := range []string{"0.3", "1.0"} {
		t.Run(version, func(t *testing.T) {
			model := &progressModel{final: make(chan struct{})}
			toolGate := make(chan struct{})
			var toolOnce, finalOnce sync.Once
			defer toolOnce.Do(func() { close(toolGate) })
			defer finalOnce.Do(func() { close(model.final) })
			clock, err := functiontool.New(functiontool.Config{Name: "GetDateTime", Description: "Inert synthetic clock"},
				func(ctx agent.Context, args map[string]any) (map[string]any, error) {
					select {
					case <-toolGate:
					case <-ctx.Done():
						return nil, ctx.Err()
					}
					return map[string]any{"time": "synthetic-only"}, nil
				})
			if err != nil {
				t.Fatal(err)
			}
			specialist, err := llmagent.New(llmagent.Config{Name: "progress_fixture", Model: model, Tools: []tool.Tool{clock}})
			if err != nil {
				t.Fatal(err)
			}
			sessions := session.InMemoryService()
			executor := native.NewKAgentExecutor(native.KAgentExecutorConfig{Stream: false, AppName: "fixture",
				RunnerConfig:   runner.Config{AppName: "fixture", Agent: specialist, SessionService: sessions},
				SessionService: sessions, SkillsDirectory: t.TempDir(), Logger: logr.Discard()})
			upstream := httptest.NewServer(old.NewJSONRPCHandler(old.NewHandler(executor)))
			defer upstream.Close()
			client, err := a2aclient.NewFromEndpoints(context.Background(), []*a2a.AgentInterface{{URL: upstream.URL, ProtocolBinding: "JSONRPC", ProtocolVersion: "0.3"}}, a2aclient.WithJSONRPCTransport(http.DefaultClient),
				a2aclient.WithCompatTransport("0.3", a2a.TransportProtocolJSONRPC, a2aclient.TransportFactoryFn(func(_ context.Context, _ *a2a.AgentCard, iface *a2a.AgentInterface) (a2aclient.Transport, error) {
					return a2av0.NewJSONRPCTransport(a2av0.JSONRPCTransportConfig{URL: iface.URL, Client: http.DefaultClient}), nil
				})))
			if err != nil {
				t.Fatal(err)
			}
			handler := NewPassthroughRequestHandler(client, nil)
			var publicURL atomic.Value
			publicURL.Store("")
			legacy, v1 := a2av0.NewJSONRPCHandler(handler), a2asrv.NewJSONRPCHandler(handler)
			proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "GET" {
					json.NewEncoder(w).Encode(map[string]any{"url": publicURL.Load(), "protocolVersion": version, "capabilities": map[string]any{"streaming": true}})
					return
				}
				w = progressWriter{w, t}
				if r.Header.Get("A2A-Version") == "0.3" {
					legacy.ServeHTTP(w, r)
				} else {
					v1.ServeHTTP(w, r)
				}
			}))
			defer proxy.Close()
			control := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/url":
					publicURL.Store(r.URL.Query().Get("url"))
				case "/tool":
					toolOnce.Do(func() { close(toolGate) })
				case "/final":
					finalOnce.Do(func() { close(model.final) })
				case "/calls":
					json.NewEncoder(w).Encode(model.calls.Load())
				default:
					http.NotFound(w, r)
				}
			}))
			defer control.Close()
			cmd := exec.Command(os.Getenv("HERMES_PYTHON"), "-m", "pytest", "tests/gateway/test_native_kagent_progress.py", "-k", "contract", "-q", "-s", "--tb=short", "-o", "addopts=")
			cmd.Dir = os.Getenv("HERMES_ROOT")
			cmd.Env = append(os.Environ(), "NATIVE_CONTROLLER_URL="+proxy.URL, "NATIVE_CONTROL_URL="+control.URL, "NATIVE_VERSION="+version)
			out, err := cmd.CombinedOutput()
			t.Log(string(out))
			if err != nil {
				t.Fatal(err)
			}
			if model.calls.Load() != 2 {
				t.Fatalf("specialist model calls=%d, want 2", model.calls.Load())
			}
		})
	}
}
