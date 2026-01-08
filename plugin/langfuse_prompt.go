//
// Tencent is pleased to support the open source community by making
// trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//
//

package plugin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/model"
)

const (
	langfusePromptAPIPath = "/api/public/v2/prompts/"

	langfuseDefaultPromptLabel = "production"

	langfusePromptTypeChat = "chat"
	langfusePromptTypeText = "text"

	langfuseChatMessageTypeChat        = "chatmessage"
	langfuseChatMessageTypePlaceholder = "placeholder"

	langfuseMaxResponseBytes = 1024 * 1024
)

var (
	errLangfuseMissingBaseURL   = errors.New("langfuse base url is empty")
	errLangfuseMissingPublicKey = errors.New("langfuse public key is empty")
	errLangfuseMissingSecretKey = errors.New("langfuse secret key is empty")

	errLangfusePromptNameEmpty = errors.New("langfuse prompt name is empty")

	errLangfuseUnsupportedPromptType = errors.New(
		"langfuse prompt type unsupported",
	)
	errLangfusePlaceholderUnsupported = errors.New(
		"langfuse placeholder message unsupported",
	)
)

// LangfusePromptRef identifies a prompt stored in Langfuse.
type LangfusePromptRef struct {
	Name string

	// Label selects the deployed label (for example, "production").
	// Only used when Version is not set.
	Label string

	// Version selects an explicit version. When set, Label is ignored.
	Version int
}

// LangfuseClientOptions configures LangfuseClient.
type LangfuseClientOptions struct {
	BaseURL   string
	PublicKey string
	SecretKey string

	HTTPClient *http.Client
}

// LangfuseClient fetches prompt templates via the Langfuse public API.
type LangfuseClient struct {
	baseURL   *url.URL
	publicKey string
	secretKey string

	httpClient *http.Client
}

// NewLangfuseClient creates a LangfuseClient.
//
// BaseURL should be a hostname or URL, for example:
//   - https://cloud.langfuse.com
//   - http://localhost:3000
//   - localhost:3000 (defaults to https)
//
// PublicKey and SecretKey are used as HTTP Basic Auth credentials.
func NewLangfuseClient(opts LangfuseClientOptions) (*LangfuseClient, error) {
	baseURL := strings.TrimSpace(opts.BaseURL)
	if baseURL == "" {
		return nil, errLangfuseMissingBaseURL
	}
	if opts.PublicKey == "" {
		return nil, errLangfuseMissingPublicKey
	}
	if opts.SecretKey == "" {
		return nil, errLangfuseMissingSecretKey
	}

	if !strings.Contains(baseURL, "://") {
		baseURL = "https://" + baseURL
	}

	parsed, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("parse langfuse base url: %w", err)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	return &LangfuseClient{
		baseURL:    parsed,
		publicKey:  opts.PublicKey,
		secretKey:  opts.SecretKey,
		httpClient: client,
	}, nil
}

// GetPrompt fetches and parses a prompt from Langfuse.
func (c *LangfuseClient) GetPrompt(
	ctx context.Context,
	ref LangfusePromptRef,
) (Prompt, error) {
	if c == nil {
		return Prompt{}, errors.New("langfuse client is nil")
	}
	name := strings.TrimSpace(ref.Name)
	if name == "" {
		return Prompt{}, errLangfusePromptNameEmpty
	}

	endpoint := c.endpointForPrompt(name)
	query := endpoint.Query()
	if ref.Version > 0 {
		query.Set("version", strconv.Itoa(ref.Version))
	} else {
		label := strings.TrimSpace(ref.Label)
		if label == "" {
			label = langfuseDefaultPromptLabel
		}
		query.Set("label", label)
	}
	endpoint.RawQuery = query.Encode()

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		endpoint.String(),
		nil,
	)
	if err != nil {
		return Prompt{}, fmt.Errorf("create request: %w", err)
	}
	req.SetBasicAuth(c.publicKey, c.secretKey)
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return Prompt{}, fmt.Errorf("langfuse request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(
		resp.Body,
		langfuseMaxResponseBytes,
	))
	if err != nil {
		return Prompt{}, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return Prompt{}, &LangfuseHTTPError{
			StatusCode: resp.StatusCode,
			Body:       strings.TrimSpace(string(body)),
		}
	}

	var parsed langfusePromptResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return Prompt{}, fmt.Errorf("decode prompt json: %w", err)
	}
	return parsed.toPrompt()
}

func (c *LangfuseClient) endpointForPrompt(name string) url.URL {
	endpoint := *c.baseURL
	endpoint.Path = endpoint.Path +
		langfusePromptAPIPath +
		url.PathEscape(name)
	return endpoint
}

// LangfuseHTTPError is returned when the Langfuse API responds with non-200.
type LangfuseHTTPError struct {
	StatusCode int
	Body       string
}

func (e *LangfuseHTTPError) Error() string {
	if e == nil {
		return "langfuse http error"
	}
	if e.Body == "" {
		return fmt.Sprintf("langfuse http %d", e.StatusCode)
	}
	return fmt.Sprintf("langfuse http %d: %s", e.StatusCode, e.Body)
}

type langfusePromptResponse struct {
	Type    string          `json:"type"`
	Name    string          `json:"name"`
	Version int             `json:"version"`
	Prompt  json.RawMessage `json:"prompt"`
}

func (r langfusePromptResponse) toPrompt() (Prompt, error) {
	switch r.Type {
	case langfusePromptTypeText:
		return r.toTextPrompt()
	case langfusePromptTypeChat:
		return r.toChatPrompt()
	default:
		return Prompt{}, fmt.Errorf(
			"%w: %q",
			errLangfuseUnsupportedPromptType,
			r.Type,
		)
	}
}

func (r langfusePromptResponse) toTextPrompt() (Prompt, error) {
	var text string
	if err := json.Unmarshal(r.Prompt, &text); err != nil {
		return Prompt{}, fmt.Errorf("decode text prompt: %w", err)
	}
	return SystemPromptWithMeta(r.Name, r.Version, text), nil
}

type langfuseChatMessage struct {
	Type    string `json:"type"`
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	Name    string `json:"name,omitempty"`
}

func (r langfusePromptResponse) toChatPrompt() (Prompt, error) {
	var msgs []langfuseChatMessage
	if err := json.Unmarshal(r.Prompt, &msgs); err != nil {
		return Prompt{}, fmt.Errorf("decode chat prompt: %w", err)
	}

	out := make([]model.Message, 0, len(msgs))
	for _, msg := range msgs {
		switch msg.Type {
		case langfuseChatMessageTypeChat:
			role, ok := parseModelRole(msg.Role)
			if !ok {
				return Prompt{}, fmt.Errorf(
					"langfuse chat role invalid: %q",
					msg.Role,
				)
			}
			out = append(out, model.Message{
				Role:    role,
				Content: msg.Content,
			})
		case langfuseChatMessageTypePlaceholder:
			return Prompt{}, fmt.Errorf(
				"%w: %q",
				errLangfusePlaceholderUnsupported,
				msg.Name,
			)
		default:
			return Prompt{}, fmt.Errorf(
				"langfuse chat message type invalid: %q",
				msg.Type,
			)
		}
	}

	prompt := Prompt{
		Name:     r.Name,
		Version:  r.Version,
		Messages: out,
	}
	if err := prompt.Validate(); err != nil {
		return Prompt{}, err
	}
	return prompt, nil
}

func parseModelRole(role string) (model.Role, bool) {
	switch role {
	case model.RoleSystem.String():
		return model.RoleSystem, true
	case model.RoleUser.String():
		return model.RoleUser, true
	case model.RoleAssistant.String():
		return model.RoleAssistant, true
	case model.RoleTool.String():
		return model.RoleTool, true
	default:
		return "", false
	}
}

// LangfusePromptResolver maps model names to Langfuse prompts.
type LangfusePromptResolver struct {
	client *LangfuseClient

	prompts       map[string]LangfusePromptRef
	defaultPrompt *LangfusePromptRef

	cacheTTL time.Duration
	now      func() time.Time

	mu    sync.RWMutex
	cache map[langfusePromptCacheKey]langfuseCachedPrompt
}

type langfusePromptCacheKey struct {
	name    string
	label   string
	version int
}

type langfuseCachedPrompt struct {
	prompt    Prompt
	expiresAt time.Time
}

// NewLangfusePromptResolver creates a model-aware resolver backed by
// Langfuse.
func NewLangfusePromptResolver(
	client *LangfuseClient,
	prompts map[string]LangfusePromptRef,
) (*LangfusePromptResolver, error) {
	if client == nil {
		return nil, errors.New("langfuse client is nil")
	}

	copied := make(map[string]LangfusePromptRef, len(prompts))
	for k, v := range prompts {
		copied[k] = v
	}

	return &LangfusePromptResolver{
		client:  client,
		prompts: copied,
		cache:   make(map[langfusePromptCacheKey]langfuseCachedPrompt),
		now:     time.Now,
	}, nil
}

// WithDefaultPrompt sets a prompt used when no model mapping exists.
func (r *LangfusePromptResolver) WithDefaultPrompt(
	ref LangfusePromptRef,
) *LangfusePromptResolver {
	if r == nil {
		return nil
	}
	r.defaultPrompt = &ref
	return r
}

// WithCacheTTL configures the in-memory cache time-to-live.
//
// Setting ttl<=0 disables caching.
func (r *LangfusePromptResolver) WithCacheTTL(
	ttl time.Duration,
) *LangfusePromptResolver {
	if r == nil {
		return nil
	}
	r.cacheTTL = ttl
	return r
}

// Resolve implements PromptResolver.
func (r *LangfusePromptResolver) Resolve(
	ctx context.Context,
	inv *agent.Invocation,
) (Prompt, bool, error) {
	if r == nil {
		return Prompt{}, false, nil
	}

	modelName := modelNameFromInvocation(inv)
	if modelName == "" {
		return Prompt{}, false, nil
	}

	ref, ok := r.prompts[modelName]
	if !ok {
		if r.defaultPrompt == nil {
			return Prompt{}, false, nil
		}
		ref = *r.defaultPrompt
	}
	ref.Name = strings.TrimSpace(ref.Name)
	if ref.Name == "" {
		return Prompt{}, false, nil
	}

	key := r.cacheKey(ref)
	if cached, ok := r.loadCache(key); ok {
		return cached, true, nil
	}

	prompt, err := r.client.GetPrompt(ctx, ref)
	if err != nil {
		return Prompt{}, false, err
	}
	r.storeCache(key, prompt)
	return prompt, true, nil
}

func (r *LangfusePromptResolver) cacheKey(
	ref LangfusePromptRef,
) langfusePromptCacheKey {
	key := langfusePromptCacheKey{name: ref.Name}
	if ref.Version > 0 {
		key.version = ref.Version
		return key
	}

	label := strings.TrimSpace(ref.Label)
	if label == "" {
		label = langfuseDefaultPromptLabel
	}
	key.label = label
	return key
}

func (r *LangfusePromptResolver) loadCache(
	key langfusePromptCacheKey,
) (Prompt, bool) {
	if r.cacheTTL <= 0 {
		return Prompt{}, false
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	entry, ok := r.cache[key]
	if !ok {
		return Prompt{}, false
	}
	if r.now().After(entry.expiresAt) {
		return Prompt{}, false
	}
	return entry.prompt, true
}

func (r *LangfusePromptResolver) storeCache(
	key langfusePromptCacheKey,
	p Prompt,
) {
	if r.cacheTTL <= 0 {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.cache[key] = langfuseCachedPrompt{
		prompt:    p,
		expiresAt: r.now().Add(r.cacheTTL),
	}
}
