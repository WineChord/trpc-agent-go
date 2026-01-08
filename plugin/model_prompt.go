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
	"errors"
	"fmt"
	"strings"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/internal/state"
	"trpc.group/trpc-go/trpc-agent-go/log"
	"trpc.group/trpc-go/trpc-agent-go/model"
)

const (
	defaultModelPromptPluginName = "model_prompt"

	langfusePromptNameKey    = "langfuse.observation.prompt.name"
	langfusePromptVersionKey = "langfuse.observation.prompt.version"
)

var errNilPromptResolver = errors.New("prompt resolver is nil")

var errModelNameEmpty = errors.New("model name is empty")

// Prompt is a resolved prompt template ready to be injected.
type Prompt struct {
	// Messages are injected into the model request.
	Messages []model.Message

	// Name is the prompt identifier from the upstream system (e.g. Langfuse).
	Name string
	// Version is the prompt version from the upstream system.
	Version int
}

// PromptResolver resolves a Prompt for the current invocation.
//
// ok=false means "no prompt for this invocation"; the plugin will do nothing.
type PromptResolver interface {
	Resolve(
		ctx context.Context,
		inv *agent.Invocation,
	) (p Prompt, ok bool, err error)
}

// PromptResolverFunc adapts a function to PromptResolver.
type PromptResolverFunc func(
	ctx context.Context,
	inv *agent.Invocation,
) (p Prompt, ok bool, err error)

// Resolve implements PromptResolver.
func (f PromptResolverFunc) Resolve(
	ctx context.Context,
	inv *agent.Invocation,
) (Prompt, bool, error) {
	if f == nil {
		return Prompt{}, false, errNilPromptResolver
	}
	return f(ctx, inv)
}

// ModelPrompt injects a resolved prompt into each model request.
//
// This plugin is designed to support model-aware system prompt switching.
// For example, when the invocation switches from "gpt-4o-mini" to "hunyuan",
// the plugin can fetch a different system prompt and prepend it to the
// request.
type ModelPrompt struct {
	name     string
	resolver PromptResolver
}

// NewModelPrompt creates a ModelPrompt plugin with a default name.
func NewModelPrompt(resolver PromptResolver) *ModelPrompt {
	return NewNamedModelPrompt(defaultModelPromptPluginName, resolver)
}

// NewNamedModelPrompt creates a ModelPrompt plugin with a custom name.
// Names must be unique per Runner.
func NewNamedModelPrompt(
	name string,
	resolver PromptResolver,
) *ModelPrompt {
	if name == "" {
		name = defaultModelPromptPluginName
	}
	return &ModelPrompt{name: name, resolver: resolver}
}

// Name implements Plugin.
func (p *ModelPrompt) Name() string { return p.name }

// Register implements Plugin.
func (p *ModelPrompt) Register(r *Registry) {
	if p == nil || r == nil {
		return
	}
	r.BeforeModel(p.beforeModel)
}

func (p *ModelPrompt) beforeModel(
	ctx context.Context,
	args *model.BeforeModelArgs,
) (*model.BeforeModelResult, error) {
	if p == nil || args == nil || args.Request == nil {
		return nil, nil
	}
	if p.resolver == nil {
		return nil, nil
	}

	inv, _ := agent.InvocationFromContext(ctx)
	prompt, ok, err := p.resolver.Resolve(ctx, inv)
	if err != nil {
		return nil, err
	}
	if !ok || len(prompt.Messages) == 0 {
		return nil, nil
	}

	injected := injectStateIntoPrompt(ctx, inv, prompt.Messages)
	applyPromptMessages(args.Request, injected)
	setPromptSpanAttributes(ctx, prompt)

	return nil, nil
}

func injectStateIntoPrompt(
	ctx context.Context,
	inv *agent.Invocation,
	msgs []model.Message,
) []model.Message {
	if inv == nil || len(msgs) == 0 {
		return msgs
	}

	out := make([]model.Message, len(msgs))
	copy(out, msgs)

	for i := range out {
		if out[i].Content == "" {
			continue
		}
		processed, err := state.InjectSessionState(
			out[i].Content,
			inv,
		)
		if err != nil {
			log.ErrorfContext(
				ctx,
				"Prompt injection state replace failed: %v",
				err,
			)
			continue
		}
		out[i].Content = processed
	}

	return out
}

func applyPromptMessages(req *model.Request, msgs []model.Message) {
	if req == nil || len(msgs) == 0 {
		return
	}

	if msgs[0].Role == model.RoleSystem {
		applyGlobalInstruction(req, msgs[0].Content)
		if len(msgs) > 1 {
			req.Messages = append(
				req.Messages[:1],
				append(msgs[1:], req.Messages[1:]...)...,
			)
		}
		return
	}

	req.Messages = append(msgs, req.Messages...)
}

func setPromptSpanAttributes(ctx context.Context, p Prompt) {
	if p.Name == "" && p.Version <= 0 {
		return
	}
	span := trace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return
	}

	attrs := make([]attribute.KeyValue, 0, 2)
	if p.Name != "" {
		attrs = append(
			attrs,
			attribute.String(langfusePromptNameKey, p.Name),
		)
	}
	if p.Version > 0 {
		attrs = append(
			attrs,
			attribute.Int(langfusePromptVersionKey, p.Version),
		)
	}
	span.SetAttributes(attrs...)
}

// StaticModelPromptResolver resolves prompts by model name.
//
// This resolver is useful for in-memory configuration or for implementing a
// watcher (for example, a Rainbow Watch client) that updates the map at
// runtime.
type StaticModelPromptResolver struct {
	mu sync.RWMutex

	prompts map[string]Prompt

	defaultPrompt *Prompt
}

// NewStaticModelPromptResolver creates a resolver from the provided map.
// The map key is model.Info().Name.
func NewStaticModelPromptResolver(
	prompts map[string]Prompt,
) *StaticModelPromptResolver {
	copied := make(map[string]Prompt, len(prompts))
	for k, v := range prompts {
		copied[k] = v
	}
	return &StaticModelPromptResolver{prompts: copied}
}

// WithDefaultPrompt sets the default prompt returned when no model match
// exists.
func (r *StaticModelPromptResolver) WithDefaultPrompt(
	p Prompt,
) *StaticModelPromptResolver {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.defaultPrompt = &p
	return r
}

// SetPrompt adds or replaces the prompt associated with the given model name.
func (r *StaticModelPromptResolver) SetPrompt(
	modelName string,
	p Prompt,
) error {
	if r == nil {
		return errors.New("resolver is nil")
	}
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return errModelNameEmpty
	}
	if err := p.Validate(); err != nil {
		return err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.prompts == nil {
		r.prompts = make(map[string]Prompt)
	}
	r.prompts[modelName] = p
	return nil
}

// DeletePrompt deletes the prompt associated with the given model name.
func (r *StaticModelPromptResolver) DeletePrompt(modelName string) bool {
	if r == nil {
		return false
	}
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return false
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.prompts == nil {
		return false
	}
	_, existed := r.prompts[modelName]
	delete(r.prompts, modelName)
	return existed
}

// Resolve implements PromptResolver.
func (r *StaticModelPromptResolver) Resolve(
	_ context.Context,
	inv *agent.Invocation,
) (Prompt, bool, error) {
	if r == nil {
		return Prompt{}, false, nil
	}

	modelName := modelNameFromInvocation(inv)
	if modelName == "" {
		return Prompt{}, false, nil
	}

	r.mu.RLock()
	p, ok := r.prompts[modelName]
	def := r.defaultPrompt
	r.mu.RUnlock()

	if ok {
		return p, true, nil
	}
	if def == nil {
		return Prompt{}, false, nil
	}
	return *def, true, nil
}

func modelNameFromInvocation(inv *agent.Invocation) string {
	if inv == nil || inv.Model == nil {
		return ""
	}
	return inv.Model.Info().Name
}

// SystemPrompt creates a Prompt containing a single system message.
func SystemPrompt(content string) Prompt {
	return Prompt{
		Messages: []model.Message{model.NewSystemMessage(content)},
	}
}

// SystemPromptWithMeta creates a system Prompt with upstream metadata.
func SystemPromptWithMeta(
	name string,
	version int,
	content string,
) Prompt {
	p := SystemPrompt(content)
	p.Name = name
	p.Version = version
	return p
}

// Validate checks basic invariants and returns a descriptive error.
func (p Prompt) Validate() error {
	if len(p.Messages) == 0 {
		return errors.New("prompt messages are empty")
	}
	for i, msg := range p.Messages {
		if !msg.Role.IsValid() {
			return fmt.Errorf("prompt message %d role invalid", i)
		}
	}
	return nil
}
