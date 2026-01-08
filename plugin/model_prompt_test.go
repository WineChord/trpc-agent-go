package plugin

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/model"
)

type stubModel struct {
	name string
}

func (m *stubModel) GenerateContent(
	context.Context,
	*model.Request,
) (<-chan *model.Response, error) {
	ch := make(chan *model.Response)
	close(ch)
	return ch, nil
}

func (m *stubModel) Info() model.Info {
	return model.Info{Name: m.name}
}

func TestModelPrompt_SystemMessageInserted(t *testing.T) {
	resolver := PromptResolverFunc(func(
		context.Context,
		*agent.Invocation,
	) (Prompt, bool, error) {
		return SystemPrompt("p1"), true, nil
	})
	mgr := MustNewManager(NewModelPrompt(resolver))

	inv := &agent.Invocation{Model: &stubModel{name: "gpt"}}
	ctx := agent.NewInvocationContext(context.Background(), inv)
	req := &model.Request{}

	cb := mgr.ModelCallbacks()
	require.NotNil(t, cb)

	_, err := cb.RunBeforeModel(ctx, &model.BeforeModelArgs{Request: req})
	require.NoError(t, err)

	require.Len(t, req.Messages, 1)
	require.Equal(t, model.RoleSystem, req.Messages[0].Role)
	require.Equal(t, "p1", req.Messages[0].Content)
}

func TestModelPrompt_SystemMessagePrepended(t *testing.T) {
	resolver := PromptResolverFunc(func(
		context.Context,
		*agent.Invocation,
	) (Prompt, bool, error) {
		return SystemPrompt("p1"), true, nil
	})
	mgr := MustNewManager(NewModelPrompt(resolver))

	inv := &agent.Invocation{Model: &stubModel{name: "gpt"}}
	ctx := agent.NewInvocationContext(context.Background(), inv)
	req := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage("orig"),
		},
	}

	cb := mgr.ModelCallbacks()
	require.NotNil(t, cb)

	_, err := cb.RunBeforeModel(ctx, &model.BeforeModelArgs{Request: req})
	require.NoError(t, err)

	require.Len(t, req.Messages, 1)
	require.Equal(t, "p1"+doubleNewline+"orig", req.Messages[0].Content)
}

func TestModelPrompt_MultiMessagePromptMerged(t *testing.T) {
	resolver := PromptResolverFunc(func(
		context.Context,
		*agent.Invocation,
	) (Prompt, bool, error) {
		return Prompt{
			Messages: []model.Message{
				model.NewSystemMessage("p1"),
				model.NewUserMessage("example"),
			},
		}, true, nil
	})
	mgr := MustNewManager(NewModelPrompt(resolver))

	inv := &agent.Invocation{Model: &stubModel{name: "gpt"}}
	ctx := agent.NewInvocationContext(context.Background(), inv)
	req := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage("orig"),
			model.NewUserMessage("hi"),
		},
	}

	cb := mgr.ModelCallbacks()
	require.NotNil(t, cb)

	_, err := cb.RunBeforeModel(ctx, &model.BeforeModelArgs{Request: req})
	require.NoError(t, err)

	require.Len(t, req.Messages, 3)
	require.Equal(t, model.RoleSystem, req.Messages[0].Role)
	require.Equal(
		t,
		"p1"+doubleNewline+"orig",
		req.Messages[0].Content,
	)
	require.Equal(t, "example", req.Messages[1].Content)
	require.Equal(t, "hi", req.Messages[2].Content)
}

func TestModelPrompt_NoMatchDoesNothing(t *testing.T) {
	resolver := PromptResolverFunc(func(
		context.Context,
		*agent.Invocation,
	) (Prompt, bool, error) {
		return Prompt{}, false, nil
	})
	mgr := MustNewManager(NewModelPrompt(resolver))

	inv := &agent.Invocation{Model: &stubModel{name: "gpt"}}
	ctx := agent.NewInvocationContext(context.Background(), inv)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("hi")},
	}

	cb := mgr.ModelCallbacks()
	require.NotNil(t, cb)

	_, err := cb.RunBeforeModel(ctx, &model.BeforeModelArgs{Request: req})
	require.NoError(t, err)

	require.Len(t, req.Messages, 1)
	require.Equal(t, model.RoleUser, req.Messages[0].Role)
}

func TestModelPrompt_ResolverErrorReturned(t *testing.T) {
	sentinel := errors.New("boom")
	resolver := PromptResolverFunc(func(
		context.Context,
		*agent.Invocation,
	) (Prompt, bool, error) {
		return Prompt{}, false, sentinel
	})
	mgr := MustNewManager(NewModelPrompt(resolver))

	inv := &agent.Invocation{Model: &stubModel{name: "gpt"}}
	ctx := agent.NewInvocationContext(context.Background(), inv)
	req := &model.Request{}

	cb := mgr.ModelCallbacks()
	require.NotNil(t, cb)

	_, err := cb.RunBeforeModel(ctx, &model.BeforeModelArgs{Request: req})
	require.Error(t, err)
	require.ErrorContains(t, err, defaultModelPromptPluginName)
	require.ErrorIs(t, err, sentinel)
}

func TestStaticModelPromptResolver_SetAndDelete(t *testing.T) {
	res := NewStaticModelPromptResolver(map[string]Prompt{
		"gpt": SystemPrompt("v1"),
	})

	inv := &agent.Invocation{Model: &stubModel{name: "gpt"}}
	got, ok, err := res.Resolve(context.Background(), inv)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "v1", got.Messages[0].Content)

	err = res.SetPrompt("gpt", SystemPrompt("v2"))
	require.NoError(t, err)

	got, ok, err = res.Resolve(context.Background(), inv)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "v2", got.Messages[0].Content)

	require.True(t, res.DeletePrompt("gpt"))
	_, ok, err = res.Resolve(context.Background(), inv)
	require.NoError(t, err)
	require.False(t, ok)
}
