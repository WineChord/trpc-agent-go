package plugin

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"trpc.group/trpc-go/trpc-agent-go/agent"
)

func TestLangfuseClient_GetPrompt_Text(t *testing.T) {
	const (
		publicKey = "pub"
		secretKey = "sec"
	)

	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			user, pass, ok := r.BasicAuth()
			require.True(t, ok)
			require.Equal(t, publicKey, user)
			require.Equal(t, secretKey, pass)

			require.Equal(
				t,
				langfusePromptAPIPath+"my-prompt",
				r.URL.Path,
			)
			require.Equal(t, "production", r.URL.Query().Get("label"))

			resp := map[string]any{
				"type":    "text",
				"name":    "my-prompt",
				"version": 7,
				"prompt":  "hello",
				"config":  map[string]any{},
				"labels":  []string{"production"},
				"tags":    []string{},
			}
			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(resp))
		},
	))
	t.Cleanup(srv.Close)

	client, err := NewLangfuseClient(LangfuseClientOptions{
		BaseURL:   srv.URL,
		PublicKey: publicKey,
		SecretKey: secretKey,
	})
	require.NoError(t, err)

	p, err := client.GetPrompt(context.Background(), LangfusePromptRef{
		Name: "my-prompt",
	})
	require.NoError(t, err)
	require.Equal(t, "my-prompt", p.Name)
	require.Equal(t, 7, p.Version)
	require.Len(t, p.Messages, 1)
	require.Equal(t, "hello", p.Messages[0].Content)
}

func TestLangfuseClient_GetPrompt_Chat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			resp := map[string]any{
				"type":    "chat",
				"name":    "p",
				"version": 3,
				"prompt": []map[string]any{
					{
						"type":    "chatmessage",
						"role":    "system",
						"content": "s",
					},
					{
						"type":    "chatmessage",
						"role":    "user",
						"content": "u",
					},
				},
				"config": map[string]any{},
				"labels": []string{"production"},
				"tags":   []string{},
			}
			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(resp))
		},
	))
	t.Cleanup(srv.Close)

	client, err := NewLangfuseClient(LangfuseClientOptions{
		BaseURL:   srv.URL,
		PublicKey: "pub",
		SecretKey: "sec",
	})
	require.NoError(t, err)

	p, err := client.GetPrompt(context.Background(), LangfusePromptRef{
		Name: "p",
	})
	require.NoError(t, err)
	require.Equal(t, "p", p.Name)
	require.Equal(t, 3, p.Version)
	require.Len(t, p.Messages, 2)
	require.Equal(t, "system", p.Messages[0].Role.String())
	require.Equal(t, "s", p.Messages[0].Content)
	require.Equal(t, "user", p.Messages[1].Role.String())
	require.Equal(t, "u", p.Messages[1].Content)
}

func TestLangfuseClient_GetPrompt_PlaceholderUnsupported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			resp := map[string]any{
				"type":    "chat",
				"name":    "p",
				"version": 3,
				"prompt": []map[string]any{
					{
						"type": "placeholder",
						"name": "other",
					},
				},
				"config": map[string]any{},
				"labels": []string{"production"},
				"tags":   []string{},
			}
			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(resp))
		},
	))
	t.Cleanup(srv.Close)

	client, err := NewLangfuseClient(LangfuseClientOptions{
		BaseURL:   srv.URL,
		PublicKey: "pub",
		SecretKey: "sec",
	})
	require.NoError(t, err)

	_, err = client.GetPrompt(context.Background(), LangfusePromptRef{
		Name: "p",
	})
	require.Error(t, err)
	require.ErrorIs(t, err, errLangfusePlaceholderUnsupported)
}

func TestLangfusePromptResolver_CacheTTL(t *testing.T) {
	var calls atomic.Int64

	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			resp := map[string]any{
				"type":    "text",
				"name":    "p",
				"version": 1,
				"prompt":  "hello",
				"config":  map[string]any{},
				"labels":  []string{"production"},
				"tags":    []string{},
			}
			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(resp))
		},
	))
	t.Cleanup(srv.Close)

	client, err := NewLangfuseClient(LangfuseClientOptions{
		BaseURL:   srv.URL,
		PublicKey: "pub",
		SecretKey: "sec",
	})
	require.NoError(t, err)

	res, err := NewLangfusePromptResolver(
		client,
		map[string]LangfusePromptRef{
			"gpt": {Name: "p"},
		},
	)
	require.NoError(t, err)
	res.WithCacheTTL(time.Minute)

	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	res.now = func() time.Time { return now }

	inv := &agent.Invocation{Model: &stubModel{name: "gpt"}}

	_, ok, err := res.Resolve(context.Background(), inv)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, int64(1), calls.Load())

	_, ok, err = res.Resolve(context.Background(), inv)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, int64(1), calls.Load())

	now = now.Add(2 * time.Minute)
	_, ok, err = res.Resolve(context.Background(), inv)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, int64(2), calls.Load())
}
