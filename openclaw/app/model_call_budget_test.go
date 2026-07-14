//
// Tencent is pleased to support the open source community by making
// trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//

package app

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/openclaw/internal/gateway"
	"trpc.group/trpc-go/trpc-agent-go/tool"
)

func TestModelCallBudgetModel_EnforcesPerContextLimit(t *testing.T) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx := withModelCallBudget(context.Background(), 2)

	_, err := wrapped.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = wrapped.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = wrapped.GenerateContent(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (2) exceeded")
	require.EqualValues(t, 2, underlying.callCount())
}

func TestModelCallBudgetModel_NoBudgetPassesThrough(t *testing.T) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)

	_, err := wrapped.GenerateContent(context.Background(), &model.Request{})
	require.NoError(t, err)
	require.EqualValues(t, 1, underlying.callCount())
}

func TestModelCallBudgetIterModel_NoBudgetPassesThrough(t *testing.T) {
	t.Parallel()

	underlying := &countingBudgetIterModel{}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)

	seq, err := iter.GenerateContentIter(context.Background(), &model.Request{})
	require.NoError(t, err)

	var responses int
	seq(func(*model.Response) bool {
		responses++
		return true
	})
	require.Equal(t, 1, responses)
	require.EqualValues(t, 1, underlying.iterCallCount())
}

func TestModelCallBudgetModel_UsesInvocationRuntimeStateFactory(
	t *testing.T,
) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	factory := newModelCallBudgetFactory(1, false, 0)
	inv := agent.NewInvocation(agent.WithInvocationRunOptions(
		agent.NewRunOptions(agent.MergeRuntimeState(map[string]any{
			modelCallBudgetRuntimeStateKey: factory,
		})),
	))
	ctx := agent.NewInvocationContext(context.Background(), inv)

	_, err := wrapped.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = wrapped.GenerateContent(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 1, underlying.callCount())
}

func TestModelCallBudgetIterModel_EnforcesPerContextLimit(t *testing.T) {
	t.Parallel()

	underlying := &countingBudgetIterModel{}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)

	ctx := withModelCallBudget(context.Background(), 1)
	_, err := iter.GenerateContentIter(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = iter.GenerateContentIter(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 1, underlying.iterCallCount())
}

func TestModelCallBudgetModel_ConcurrentCallsShareLimit(t *testing.T) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx := withModelCallBudget(context.Background(), 3)

	var wg sync.WaitGroup
	var successes atomic.Int64
	var failures atomic.Int64
	errs := make(chan string, 16)
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := wrapped.GenerateContent(ctx, &model.Request{})
			if err != nil {
				stopErr, ok := agent.AsStopError(err)
				if !ok || stopErr.Message !=
					"max LLM calls (3) exceeded" {
					errs <- err.Error()
				}
				failures.Add(1)
				return
			}
			successes.Add(1)
		}()
	}
	wg.Wait()
	close(errs)

	for msg := range errs {
		require.Empty(t, msg)
	}
	require.EqualValues(t, 3, successes.Load())
	require.EqualValues(t, 13, failures.Load())
	require.EqualValues(t, 3, underlying.callCount())
}

func TestModelCallBudgetModel_InfoPassesThrough(t *testing.T) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)

	require.Equal(t, underlying.Info(), wrapped.Info())
}

func TestNewModelCallBudgetModel_Nil(t *testing.T) {
	t.Parallel()

	require.Nil(t, newModelCallBudgetModel(nil))
}

func TestModelCallBudget_Guards(t *testing.T) {
	t.Parallel()

	require.Nil(t, newModelCallBudget(0, false, 0))
	require.Nil(t, newModelCallBudget(-1, false, 0))
	require.NotNil(t, newModelCallBudget(0, false, time.Second))
	_, err := (*modelCallBudget)(nil).use(context.Background())
	require.NoError(t, err)

	ctx := withModelCallBudget(nil, 1)
	require.NotNil(t, ctx)
	require.NotNil(t, modelCallBudgetFromContext(ctx))

	inv := agent.NewInvocation(agent.WithInvocationRunOptions(
		agent.NewRunOptions(agent.MergeRuntimeState(map[string]any{
			modelCallBudgetRuntimeStateKey: "not-a-budget",
		})),
	))
	ctx = agent.NewInvocationContext(context.Background(), inv)
	require.Nil(t, modelCallBudgetFromContext(ctx))
}

func TestModelCallBudget_FinalRequestEvidenceGuards(t *testing.T) {
	t.Parallel()

	var nilBudget *modelCallBudget
	require.Equal(t, modelCallBudgetFinalRequestConfig{}, nilBudget.finalConfig())
	require.False(t, modelCallBudgetHasToolEvidence(nil))
	require.True(t, modelCallBudgetHasToolEvidence(&model.Request{
		Messages: []model.Message{{Role: model.RoleAssistant, ToolID: "call_1"}},
	}))
	require.True(t, modelCallBudgetHasToolEvidence(&model.Request{
		Messages: []model.Message{{
			Role: model.RoleAssistant,
			ToolCalls: []model.ToolCall{{
				ID:   "call_1",
				Type: "function",
			}},
		}},
	}))

	direct := nilBudget.applyFinalRequest(&model.Request{
		Messages: []model.Message{
			model.NewToolMessage("call_1", "search", "evidence"),
		},
	})
	require.NotNil(t, direct)
	require.Contains(t, budgetTestMessageText(direct.Messages), "final allowed")

	budget := newModelCallBudget(2, true, 0)
	budget.rememberRequest(&model.Request{
		Messages: []model.Message{
			model.NewUserMessage("question"),
			model.NewToolMessage("call_1", "search", "original evidence"),
		},
	})
	budget.rememberRequest(&model.Request{
		Messages: []model.Message{
			model.NewToolMessage("call_2", "search", "shorter evidence"),
		},
	})
	finalReq := budget.applyFinalRequest(&model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
	})
	content := budgetTestMessageText(finalReq.Messages)
	require.Contains(t, content, "original evidence")
	require.NotContains(t, content, "shorter evidence")
}

func TestModelCallBudgetDeadlineSoon_Guards(t *testing.T) {
	t.Parallel()

	require.False(t, modelCallBudgetDeadlineSoon(nil, time.Second))
	require.False(
		t,
		modelCallBudgetDeadlineSoon(context.Background(), time.Second),
	)

	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	require.False(t, modelCallBudgetDeadlineSoon(ctx, 0))
	require.True(t, modelCallBudgetDeadlineSoon(ctx, time.Minute))
}

func TestModelCallBudgetPrefinalHelpers_Guards(t *testing.T) {
	t.Parallel()

	ctx, cancel, ok := modelCallBudgetPrefinalContext(
		context.Background(),
		time.Second,
	)
	require.False(t, ok)
	require.Nil(t, cancel)
	require.NotNil(t, ctx)

	nearDeadline, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(10*time.Millisecond),
	)
	defer cancel()
	ctx, prefinalCancel, ok := modelCallBudgetPrefinalContext(
		nearDeadline,
		time.Minute,
	)
	require.False(t, ok)
	require.Nil(t, prefinalCancel)
	require.Same(t, nearDeadline, ctx)

	require.False(t, modelCallBudgetPrefinalTimedOut(nil, context.Background()))
	require.False(t, modelCallBudgetTimeoutResponse(
		timeoutResponse(time.Second, context.DeadlineExceeded),
		context.Background(),
		context.Background(),
		&modelCallBudget{deadlineWindow: time.Second},
	))

	canceled, stop := context.WithCancel(context.Background())
	stop()
	require.False(t, modelCallBudgetSendResponse(
		canceled,
		make(chan *model.Response),
		&model.Response{},
	))
}

func TestModelCallBudgetCallbacks_RunBeforeModel(t *testing.T) {
	t.Parallel()

	callbacks := modelCallBudgetCallbacks()
	require.NotNil(t, callbacks)
	require.Len(t, callbacks.BeforeModel, 1)

	result, err := callbacks.RunBeforeModel(
		context.Background(),
		&model.BeforeModelArgs{Request: &model.Request{}},
	)
	require.NoError(t, err)
	require.Nil(t, result)
}

func TestBaseLLMAgentOptions_AddsModelCallBudgetCallbacks(t *testing.T) {
	t.Parallel()

	withoutBudget := baseLLMAgentOptions(
		&countingBudgetModel{},
		agentConfig{},
		"",
		"",
		model.GenerationConfig{},
		nil,
	)
	withBudget := baseLLMAgentOptions(
		&countingBudgetModel{},
		agentConfig{MaxLLMCalls: 1},
		"",
		"",
		model.GenerationConfig{},
		nil,
	)

	require.Len(t, withBudget, len(withoutBudget)+1)
}

func TestAppendModelCallBudgetGatewayOption_Disabled(t *testing.T) {
	t.Parallel()

	opts := appendModelCallBudgetGatewayOption(nil, 0, false, 0)
	require.Empty(t, opts)
}

func TestAppendModelCallBudgetGatewayOption_AddsRunBudget(t *testing.T) {
	t.Parallel()

	opts := appendModelCallBudgetGatewayOption(nil, 1, false, 0)
	require.Len(t, opts, 1)
}

func TestAppendModelCallBudgetGatewayOption_AddsDeadlineBudget(
	t *testing.T,
) {
	t.Parallel()

	opts := appendModelCallBudgetGatewayOption(nil, 0, false, time.Minute)
	require.Len(t, opts, 1)
}

func TestModelCallBudgetRunOptions(t *testing.T) {
	t.Parallel()

	require.Nil(t, modelCallBudgetRunOptions(0, false, 0))

	runOpts := modelCallBudgetRunOptions(0, false, time.Minute)
	require.Len(t, runOpts, 1)
	opts := agent.NewRunOptions(runOpts...)
	factory, ok := opts.RuntimeState[modelCallBudgetRuntimeStateKey].(*modelCallBudgetFactory)
	require.True(t, ok)
	require.Equal(t, time.Minute, factory.deadlineWindow)
}

func TestModelCallBudgetModel_FinalizesOnLastAllowedCall(t *testing.T) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx := withModelCallBudgetValue(
		context.Background(),
		newModelCallBudget(1, true, 0),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
		GenerationConfig: model.GenerationConfig{
			Stream: true,
		},
		ExtraFields: map[string]any{
			"parallel_tool_calls": true,
			"response_format":     "json",
			"tool_choice":         "required",
			"tools":               []string{"search"},
		},
	}

	_, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)

	got := underlying.lastRequest()
	require.NotNil(t, got)
	require.Nil(t, got.Tools)
	require.Len(t, got.Messages, 2)
	require.Contains(
		t,
		got.Messages[1].Content,
		"final allowed model call",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"Do not emit tool calls",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"<tool_call>",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"visible assistant message content",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"not only in internal reasoning",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"Do not describe plans or next steps",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"answer now with the best supported final value",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"FINAL ANSWER:",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"avoid extra explanation",
	)
	require.Nil(t, req.Tools)
	require.Len(t, req.Messages, 2)
	require.Equal(t, map[string]any{
		"response_format": "json",
	}, req.ExtraFields)
	require.Equal(t, map[string]any{
		"response_format": "json",
	}, got.ExtraFields)
	require.False(t, got.Stream)
	require.False(t, req.Stream)
}

func TestModelCallBudgetModel_FinalizesWithStoredEvidenceMessages(
	t *testing.T,
) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx := withModelCallBudgetValue(
		context.Background(),
		newModelCallBudget(2, true, 0),
	)
	richReq := &model.Request{
		Messages: []model.Message{
			model.NewUserMessage("question"),
			model.NewAssistantMessage("I should verify the candidate."),
			model.NewToolMessage(
				"call_1",
				"web_fetch",
				"Michele Fitzgerald was born May 5, 1990.",
			),
		},
		Tools: map[string]tool.Tool{"web_fetch": nil},
	}
	_, err := wrapped.GenerateContent(ctx, richReq)
	require.NoError(t, err)

	minimalReq := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage("final system"),
			model.NewUserMessage("question"),
		},
		Tools: map[string]tool.Tool{"web_fetch": nil},
	}
	_, err = wrapped.GenerateContent(ctx, minimalReq)
	require.NoError(t, err)

	got := underlying.lastRequest()
	require.NotNil(t, got)
	require.Nil(t, got.Tools)
	require.Len(t, got.Messages, 4)
	require.Equal(t, model.RoleTool, got.Messages[2].Role)
	require.Contains(t, got.Messages[2].Content, "Michele Fitzgerald")
	require.Contains(
		t,
		got.Messages[3].Content,
		"final allowed model call",
	)
	require.Len(t, minimalReq.Messages, 2)
	require.Len(t, minimalReq.Tools, 1)
}

func TestModelCallBudgetModel_FinalizationDisablesThinking(
	t *testing.T,
) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx := withModelCallBudgetValue(
		context.Background(),
		newModelCallBudget(
			1,
			true,
			0,
			modelCallBudgetFinalRequestConfig{DisableThinking: true},
		),
	)
	req := &model.Request{
		GenerationConfig: model.GenerationConfig{
			ThinkingEnabled: model.BoolPtr(true),
		},
		Messages: []model.Message{model.NewUserMessage("question")},
	}

	_, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)

	got := underlying.lastRequest()
	require.NotNil(t, got)
	require.NotNil(t, got.ThinkingEnabled)
	require.False(t, *got.ThinkingEnabled)
	require.NotNil(t, req.ThinkingEnabled)
	require.False(t, *req.ThinkingEnabled)
}

func TestFinalModelCallRequest_DropsReasoningWhenConfigured(t *testing.T) {
	t.Parallel()

	req := &model.Request{
		Messages: []model.Message{
			{
				Role:               model.RoleAssistant,
				Content:            "visible answer",
				ReasoningContent:   "private reasoning",
				ReasoningSignature: "signature",
			},
		},
	}

	got := finalModelCallRequest(
		req,
		modelCallBudgetFinalRequestConfig{DropReasoningContent: true},
	)

	require.Len(t, got.Messages, 2)
	require.Equal(t, "visible answer", got.Messages[0].Content)
	require.Empty(t, got.Messages[0].ReasoningContent)
	require.Empty(t, got.Messages[0].ReasoningSignature)
	require.Equal(t, "private reasoning", req.Messages[0].ReasoningContent)
	require.Equal(t, "signature", req.Messages[0].ReasoningSignature)
}

func TestFinalModelCallRequest_DropsSkillOverviewSystemPrompt(
	t *testing.T,
) {
	t.Parallel()

	req := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage(
				"\n" + finalModelCallSkillOverviewPrefix +
					"\n\n- skill-a: " + strings.Repeat("x", 200),
			),
			model.NewSystemMessage("memory and stable instructions"),
			model.NewUserMessage("latest question"),
		},
	}

	got := finalModelCallRequest(
		req,
		modelCallBudgetFinalRequestConfig{MaxInputTokens: 1000},
	)

	content := budgetTestMessageText(got.Messages)
	require.NotContains(t, content, "skill-a")
	require.Contains(t, content, "memory and stable instructions")
	require.Contains(t, content, "latest question")
	require.Contains(t, content, "final allowed model call")
}

func TestFinalModelCallRequest_TrimsContextWhenConfigured(t *testing.T) {
	t.Parallel()

	req := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage("system instructions"),
			model.NewUserMessage("old question " + strings.Repeat("x", 800)),
			model.NewAssistantMessage(
				"old answer " + strings.Repeat("y", 800),
			),
			model.NewUserMessage("latest question"),
			model.NewToolMessage("call_1", "search", "latest evidence"),
		},
		Tools: map[string]tool.Tool{"search": nil},
	}

	got := finalModelCallRequest(
		req,
		modelCallBudgetFinalRequestConfig{MaxInputTokens: 20},
	)

	require.Nil(t, got.Tools)
	require.Less(t, len(got.Messages), len(req.Messages)+1)
	require.Equal(t, model.RoleSystem, got.Messages[0].Role)
	require.Equal(t, model.RoleUser, got.Messages[len(got.Messages)-1].Role)
	require.Contains(
		t,
		got.Messages[len(got.Messages)-1].Content,
		"final allowed model call",
	)
	content := budgetTestMessageText(got.Messages)
	require.Contains(t, content, "latest question")
	require.Contains(t, content, "latest evidence")
	require.NotContains(t, content, "old question")
	require.NotContains(t, content, "old answer")
}

func TestFinalModelCallRequest_UsesConfiguredTokenEstimate(t *testing.T) {
	t.Parallel()

	req := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage("system instructions"),
			model.NewUserMessage(
				"old question " + strings.Repeat("x", 100),
			),
			model.NewAssistantMessage(
				"old answer " + strings.Repeat("y", 100),
			),
			model.NewUserMessage("latest question"),
		},
	}

	relaxed := finalModelCallRequest(
		req,
		modelCallBudgetFinalRequestConfig{MaxInputTokens: 80},
	)
	strict := finalModelCallRequest(
		req,
		modelCallBudgetFinalRequestConfig{
			MaxInputTokens:      80,
			ApproxRunesPerToken: 1,
		},
	)

	relaxedContent := budgetTestMessageText(relaxed.Messages)
	require.Contains(t, relaxedContent, "old question")
	require.Contains(t, relaxedContent, "old answer")

	strictContent := budgetTestMessageText(strict.Messages)
	require.Contains(t, strictContent, "latest question")
	require.NotContains(t, strictContent, "old question")
	require.NotContains(t, strictContent, "old answer")
}

func TestFinalModelCallRequest_TrimsSingleUserToolChain(t *testing.T) {
	t.Parallel()

	req := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage(
				"system " + strings.Repeat("s", 600),
			),
			model.NewUserMessage(
				"solve the task " + strings.Repeat("q", 120),
			),
		},
	}
	for i := 0; i < 16; i++ {
		req.Messages = append(req.Messages, model.Message{
			Role: model.RoleAssistant,
			ToolCalls: []model.ToolCall{{
				ID:   fmt.Sprintf("call_%02d", i),
				Type: "function",
				Function: model.FunctionDefinitionParam{
					Name:      "search",
					Arguments: []byte(`{"query":"large query payload"}`),
				},
			}},
		})
		req.Messages = append(req.Messages, model.NewToolMessage(
			fmt.Sprintf("call_%02d", i),
			"search",
			fmt.Sprintf(
				"tool-result-%02d %s",
				i,
				strings.Repeat("r", 120),
			),
		))
	}

	got := finalModelCallRequest(
		req,
		modelCallBudgetFinalRequestConfig{
			MaxInputTokens:      1600,
			ApproxRunesPerToken: 1,
		},
	)

	require.Less(t, len(got.Messages), len(req.Messages))
	content := budgetTestMessageText(got.Messages)
	require.Contains(t, content, "solve the task")
	require.Contains(t, content, "tool-result-15")
	require.NotContains(t, content, "tool-result-00")
	for _, msg := range got.Messages {
		require.NotEqual(t, model.RoleTool, msg.Role)
		require.Empty(t, msg.ToolCalls)
	}
}

func TestFinalModelCallRequest_PreservesAnswerFormatInstruction(
	t *testing.T,
) {
	t.Parallel()

	userPrompt := "Solve the visible task.\n\n" +
		strings.Repeat("background evidence ", 120) +
		"\n\nFINAL ANSWER: put only the numeric value on the final line." +
		"\n\n" + strings.Repeat("attachment paths and metadata ", 120)
	req := &model.Request{
		Messages: []model.Message{
			model.NewSystemMessage("system instructions"),
			model.NewUserMessage(userPrompt),
			model.NewAssistantMessage("I will inspect the evidence."),
			model.NewToolMessage("call_1", "image_inspect", "evidence"),
		},
	}

	got := finalModelCallRequest(
		req,
		modelCallBudgetFinalRequestConfig{
			MaxInputTokens:      1000,
			ApproxRunesPerToken: 1,
		},
	)

	content := budgetTestMessageText(got.Messages)
	require.Contains(t, content, "FINAL ANSWER:")
	require.Contains(t, content, "numeric value")
	require.Contains(t, content, "evidence")
	require.Contains(t, content, "final allowed model call")
}

func TestFinalModelCallHelpers_EdgeCases(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	counter := model.NewSimpleTokenCounter(model.WithApproxRunesPerToken(1))
	require.Empty(t, finalModelCallRecentTailEvidenceSnippet([]model.Message{
		model.NewSystemMessage("system only"),
	}))
	require.Len(
		t,
		finalModelCallEvidenceSnippet(strings.Repeat("x", 80)),
		finalModelCallEvidenceSnippetLen,
	)
	require.Equal(t, 0, finalModelCallTrimBudget(ctx, counter, 0))
	require.Equal(
		t,
		20,
		finalModelCallTrimBudget(ctx, failingBudgetTokenCounter{}, 20),
	)
	require.False(t, finalModelCallFits(ctx, counter, nil, 20))
	require.False(t, finalModelCallFits(ctx, counter, []model.Message{
		model.NewUserMessage("question"),
	}, 0))
	require.Nil(t, finalModelCallTailEvidenceMessages(
		ctx,
		counter,
		[]model.Message{model.NewSystemMessage("system only")},
		20,
	))

	oversizedPrefix := []model.Message{
		model.NewUserMessage(strings.Repeat("x", 200)),
	}
	require.Equal(t, oversizedPrefix, finalModelCallTailEvidenceWithPrefix(
		ctx,
		counter,
		[]model.Message{
			model.NewUserMessage("question"),
			model.NewAssistantMessage("answer"),
		},
		oversizedPrefix,
		0,
		1,
	))

	compact := finalModelCallCompactTailEvidenceWithPrefix(
		ctx,
		counter,
		[]model.Message{
			model.NewUserMessage("question"),
			model.NewAssistantMessage(strings.Repeat("evidence ", 80)),
		},
		[]model.Message{model.NewUserMessage("question")},
		0,
		80,
	)
	require.NotEmpty(t, compact)
	require.Contains(t, budgetTestMessageText(compact), "evidence")

	compacted := finalModelCallCompactMessages(
		[]model.Message{model.NewUserMessage(strings.Repeat("abc", 20))},
		12,
	)
	require.Len(t, compacted, 1)
	require.LessOrEqual(t, len([]rune(compacted[0].Content)), 12)

	require.Equal(
		t,
		"[Tool result: tool]",
		finalModelCallToolResultText(model.Message{}),
	)
	require.Equal(t, 1, finalModelCallAnchorUserIndex(
		[]model.Message{
			model.NewSystemMessage("system"),
			model.NewAssistantMessage("assistant"),
		},
		-1,
	))
	require.Equal(t, -1, finalModelCallAnchorUserIndex(
		[]model.Message{model.NewSystemMessage("system")},
		0,
	))
	require.Equal(t, 0, finalModelCallPartRuneLimit(0, 2))
	require.Equal(t, 10, finalModelCallPartRuneLimit(10, 0))
	require.Equal(t, 1, finalModelCallPartRuneLimit(1, 10))
	require.Equal(t, "\n\nFIN", finalModelCallTrimContent(
		"FINAL ANSWER: value",
		5,
	))
	require.Empty(t, finalModelCallTrimContentPlain("abc", 0))
	require.Equal(t, 0, finalModelCallParagraphStart("answer", 0))
	require.Equal(t, 2, finalModelCallParagraphStart("a\nanswer", 4))
	require.Equal(t, 6, finalModelCallParagraphEnd("answer", len("answer")))
	require.Equal(t, 6, finalModelCallParagraphEnd("answer\nnext", 0))
	require.Equal(
		t,
		"xxxxx",
		finalModelCallLimitSnippet(strings.Repeat("x", 20), -1, 5),
	)
	require.Equal(
		t,
		"xxxxx",
		finalModelCallLimitSnippet(strings.Repeat("x", 20), 100, 5),
	)
	require.Equal(t, []model.Message{model.NewUserMessage("tail")},
		finalModelCallNormalizeTail([]model.Message{
			model.NewToolMessage("call_1", "search", "skip"),
			model.NewUserMessage("tail"),
		}),
	)
	require.NotNil(t, applyFinalModelCallRequest(nil))
}

func budgetTestMessageText(messages []model.Message) string {
	var b strings.Builder
	for _, msg := range messages {
		b.WriteString(msg.Content)
		b.WriteByte('\n')
	}
	return b.String()
}

func TestModelCallBudgetModel_FinalizesNearDeadline(t *testing.T) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, time.Minute),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
		GenerationConfig: model.GenerationConfig{
			Stream: true,
		},
	}

	_, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)

	got := underlying.lastRequest()
	require.NotNil(t, got)
	require.Nil(t, got.Tools)
	require.Len(t, got.Messages, 2)
	require.Contains(
		t,
		got.Messages[1].Content,
		"final allowed model call",
	)
	require.Nil(t, req.Tools)
	require.Len(t, req.Messages, 2)
	require.False(t, got.Stream)
	require.False(t, req.Stream)
}

func TestModelCallBudgetIterModel_FinalizesNearDeadline(t *testing.T) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, time.Minute),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
		GenerationConfig: model.GenerationConfig{
			Stream: true,
		},
	}

	_, err := iter.GenerateContentIter(ctx, req)
	require.NoError(t, err)

	got := underlying.lastRequest()
	require.NotNil(t, got)
	require.Nil(t, underlying.lastIterRequest())
	require.Nil(t, got.Tools)
	require.Len(t, got.Messages, 2)
	require.Contains(
		t,
		got.Messages[1].Content,
		"final allowed model call",
	)
	require.Nil(t, req.Tools)
	require.Len(t, req.Messages, 2)
	require.False(t, got.Stream)
	require.False(t, req.Stream)
}

func TestModelCallBudgetModel_FinalizesWhenPrefinalWindowExpires(
	t *testing.T,
) {
	t.Parallel()

	const (
		prefinalTestDeadline = 500 * time.Millisecond
		prefinalTestWindow   = 300 * time.Millisecond
	)

	underlying := &prefinalTimeoutBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(prefinalTestDeadline),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, prefinalTestWindow),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	ch, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)
	var got []*model.Response
	for resp := range ch {
		got = append(got, resp)
	}

	require.Len(t, got, 1)
	require.Equal(t, "final answer", got[0].Choices[0].Message.Content)
	requests := underlying.requestsSnapshot()
	require.Len(t, requests, 2)
	require.NotNil(t, requests[0].Tools)
	require.Nil(t, requests[1].Tools)
	require.Len(t, requests[1].Messages, 2)
	require.Contains(
		t,
		requests[1].Messages[1].Content,
		"final allowed model call",
	)
	require.Nil(t, req.Tools)
}

func TestModelCallBudgetIterModel_FinalizesWhenPrefinalWindowExpires(
	t *testing.T,
) {
	t.Parallel()

	const (
		prefinalTestDeadline = 500 * time.Millisecond
		prefinalTestWindow   = 300 * time.Millisecond
	)

	underlying := &prefinalTimeoutBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(prefinalTestDeadline),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, prefinalTestWindow),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	seq, err := iter.GenerateContentIter(ctx, req)
	require.NoError(t, err)
	var got []*model.Response
	seq(func(resp *model.Response) bool {
		got = append(got, resp)
		return true
	})

	require.Len(t, got, 1)
	require.Equal(t, "final answer", got[0].Choices[0].Message.Content)
	requests := underlying.requestsSnapshot()
	require.Len(t, requests, 2)
	require.Empty(t, underlying.iterRequestsSnapshot())
	require.NotNil(t, requests[0].Tools)
	require.Nil(t, requests[1].Tools)
	require.Len(t, requests[1].Messages, 2)
	require.Contains(
		t,
		requests[1].Messages[1].Content,
		"final allowed model call",
	)
	require.Nil(t, req.Tools)
}

func TestModelCallBudgetModel_FinalizesWhenPrefinalProviderStalls(
	t *testing.T,
) {
	t.Parallel()

	underlying := newStalledPrefinalBudgetModel()
	t.Cleanup(underlying.release)
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(800*time.Millisecond),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 600*time.Millisecond),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	ch, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)
	select {
	case resp := <-ch:
		require.Equal(
			t,
			"final answer",
			resp.Choices[0].Message.Content,
		)
	case <-time.After(600 * time.Millisecond):
		t.Fatal("prefinal provider stall blocked finalization")
	}
	require.Equal(t, int64(2), underlying.callCount())
}

func TestModelCallBudgetIterModel_FinalizesWhenPrefinalProviderStalls(
	t *testing.T,
) {
	t.Parallel()

	underlying := newStalledPrefinalBudgetModel()
	t.Cleanup(underlying.release)
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(800*time.Millisecond),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 600*time.Millisecond),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	seq, err := iter.GenerateContentIter(ctx, req)
	require.NoError(t, err)
	result := make(chan *model.Response, 1)
	go seq(func(resp *model.Response) bool {
		result <- resp
		return true
	})
	select {
	case resp := <-result:
		require.Equal(
			t,
			"final answer",
			resp.Choices[0].Message.Content,
		)
	case <-time.After(600 * time.Millisecond):
		t.Fatal("prefinal iter provider stall blocked finalization")
	}
	require.Equal(t, int64(2), underlying.callCount())
	require.Zero(t, underlying.iterCallCount())
}

func TestModelCallBudgetModel_SnapshotsRequestForStalledProvider(
	t *testing.T,
) {
	t.Parallel()

	underlying := &retainingPrefinalBudgetModel{
		observed: make(chan retainedBudgetRequest, 1),
	}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(800*time.Millisecond),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 600*time.Millisecond),
	)
	req := modelCallBudgetTestRequest()
	req.Stream = true

	ch, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)
	resp := <-ch
	require.Equal(t, "final answer", resp.Choices[0].Message.Content)
	select {
	case got := <-underlying.observed:
		require.True(t, got.hasTools)
		require.Equal(t, 1, got.messages)
		require.True(t, got.stream)
	case <-time.After(600 * time.Millisecond):
		t.Fatal("stalled provider did not inspect its request snapshot")
	}
}

func TestModelCallBudgetModel_StopsAfterTerminalProviderResponse(
	t *testing.T,
) {
	t.Parallel()

	underlying := &scriptedPrefinalBudgetModel{
		initial: []*model.Response{modelCallBudgetTestFinalResponse()},
	}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 500*time.Millisecond),
	)

	ch, err := wrapped.GenerateContent(ctx, modelCallBudgetTestRequest())
	require.NoError(t, err)
	result := make(chan []*model.Response, 1)
	go func() {
		var responses []*model.Response
		for resp := range ch {
			responses = append(responses, resp)
		}
		result <- responses
	}()
	var responses []*model.Response
	select {
	case responses = <-result:
	case <-time.After(300 * time.Millisecond):
		t.Fatal("terminal response did not stop provider forwarding")
	}

	require.Len(t, responses, 1)
	require.True(t, responses[0].Done)
	require.Equal(t, int64(1), underlying.callCount())
}

func TestModelCallBudgetIterModel_StopsAfterTerminalProviderResponse(
	t *testing.T,
) {
	t.Parallel()

	underlying := &scriptedPrefinalBudgetModel{
		initial: []*model.Response{modelCallBudgetTestFinalResponse()},
	}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 500*time.Millisecond),
	)

	seq, err := iter.GenerateContentIter(ctx, modelCallBudgetTestRequest())
	require.NoError(t, err)
	result := make(chan []*model.Response, 1)
	go func() {
		var responses []*model.Response
		seq(func(resp *model.Response) bool {
			responses = append(responses, resp)
			return true
		})
		result <- responses
	}()
	var responses []*model.Response
	select {
	case responses = <-result:
	case <-time.After(300 * time.Millisecond):
		t.Fatal("terminal response did not stop sequence forwarding")
	}

	require.Len(t, responses, 1)
	require.True(t, responses[0].Done)
	require.Equal(t, int64(1), underlying.callCount())
	require.Zero(t, underlying.iterCallCount())
}

func TestModelCallBudgetIterModel_PropagatesYieldFalse(t *testing.T) {
	t.Parallel()

	underlying := &scriptedPrefinalBudgetModel{
		initial: []*model.Response{
			modelCallBudgetTestResponse("first", false),
			modelCallBudgetTestResponse("second", false),
		},
	}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 500*time.Millisecond),
	)

	seq, err := iter.GenerateContentIter(ctx, modelCallBudgetTestRequest())
	require.NoError(t, err)
	var responses []*model.Response
	seq(func(resp *model.Response) bool {
		responses = append(responses, resp)
		return false
	})

	require.Len(t, responses, 1)
	require.Equal(t, "first", responses[0].Choices[0].Message.Content)
	require.Equal(t, int64(1), underlying.callCount())
	require.Zero(t, underlying.iterCallCount())
}

func TestModelCallBudgetForwardResponses_PreservesConsumerStopAtDeadline(
	t *testing.T,
) {
	t.Parallel()

	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	prefinalCtx, prefinalCancel := context.WithDeadline(
		ctx,
		time.Now().Add(20*time.Millisecond),
	)
	defer prefinalCancel()
	responses := make(chan *model.Response, 1)
	responses <- modelCallBudgetTestResponse("partial", false)

	result := modelCallBudgetForwardResponses(
		ctx,
		prefinalCtx,
		responses,
		&modelCallBudget{deadlineWindow: time.Second},
		func(*model.Response) modelCallBudgetForwardResult {
			<-prefinalCtx.Done()
			return modelCallBudgetForwardResult{stopped: true}
		},
	)

	require.True(t, result.stopped)
	require.False(t, result.timedOut)
}

func TestModelCallBudgetModel_FinalizesTerminalTimeoutOnOpenChannel(
	t *testing.T,
) {
	t.Parallel()

	finalCanceled := make(chan struct{})
	underlying := &scriptedPrefinalBudgetModel{
		initial: []*model.Response{
			timeoutResponse(time.Minute, context.DeadlineExceeded),
		},
		finalCanceled: finalCanceled,
	}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(2*time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, time.Second),
	)

	ch, err := wrapped.GenerateContent(ctx, modelCallBudgetTestRequest())
	require.NoError(t, err)
	select {
	case resp := <-ch:
		require.Equal(
			t,
			"final answer",
			resp.Choices[0].Message.Content,
		)
	case <-time.After(300 * time.Millisecond):
		t.Fatal("terminal timeout waited for the prefinal deadline")
	}
	select {
	case <-finalCanceled:
	case <-time.After(300 * time.Millisecond):
		t.Fatal("terminal final response did not cancel the provider")
	}
	require.Equal(t, int64(2), underlying.callCount())
	require.False(t, underlying.finalStartedBeforeCancel())
}

func TestModelCallBudgetIterModel_CancelsFinalProvider(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name  string
		yield bool
	}{
		{name: "terminal delivered", yield: true},
		{name: "consumer stopped", yield: false},
	} {
		testCase := tt
		t.Run(testCase.name, func(t *testing.T) {
			finalCanceled := make(chan struct{})
			underlying := &scriptedPrefinalBudgetModel{
				finalCanceled: finalCanceled,
			}
			wrapped := newModelCallBudgetModel(underlying)
			iter, ok := wrapped.(model.IterModel)
			require.True(t, ok)
			ctx, cancel := context.WithDeadline(
				context.Background(),
				time.Now().Add(time.Second),
			)
			defer cancel()
			ctx = withModelCallBudgetValue(
				ctx,
				newModelCallBudget(0, false, time.Minute),
			)

			seq, err := iter.GenerateContentIter(
				ctx,
				modelCallBudgetTestRequest(),
			)
			require.NoError(t, err)
			seq(func(*model.Response) bool { return testCase.yield })
			select {
			case <-finalCanceled:
			case <-time.After(300 * time.Millisecond):
				t.Fatal("final provider context was not canceled")
			}
			require.Equal(t, int64(1), underlying.callCount())
			require.Zero(t, underlying.iterCallCount())
		})
	}
}

func TestModelCallBudgetIterModel_CancelsFallbackProvider(
	t *testing.T,
) {
	t.Parallel()

	for _, tt := range []struct {
		name  string
		yield bool
	}{
		{name: "terminal delivered", yield: true},
		{name: "consumer stopped", yield: false},
	} {
		testCase := tt
		t.Run(testCase.name, func(t *testing.T) {
			finalCanceled := make(chan struct{})
			underlying := &scriptedPrefinalBudgetModel{
				initial: []*model.Response{
					timeoutResponse(
						time.Minute,
						context.DeadlineExceeded,
					),
				},
				finalCanceled: finalCanceled,
			}
			wrapped := newModelCallBudgetModel(underlying)
			iter, ok := wrapped.(model.IterModel)
			require.True(t, ok)
			ctx, cancel := context.WithDeadline(
				context.Background(),
				time.Now().Add(2*time.Second),
			)
			defer cancel()
			ctx = withModelCallBudgetValue(
				ctx,
				newModelCallBudget(0, false, time.Second),
			)

			seq, err := iter.GenerateContentIter(
				ctx,
				modelCallBudgetTestRequest(),
			)
			require.NoError(t, err)
			seq(func(resp *model.Response) bool {
				require.Equal(
					t,
					"final answer",
					resp.Choices[0].Message.Content,
				)
				return testCase.yield
			})
			select {
			case <-finalCanceled:
			case <-time.After(300 * time.Millisecond):
				t.Fatal("fallback provider context was not canceled")
			}
			require.Equal(t, int64(2), underlying.callCount())
			require.Zero(t, underlying.iterCallCount())
			require.False(
				t,
				underlying.finalStartedBeforeCancel(),
			)
		})
	}
}

func TestModelCallBudgetModel_FinalizesWhenOutputIsBackpressured(
	t *testing.T,
) {
	t.Parallel()

	underlying := &scriptedPrefinalBudgetModel{
		initial: []*model.Response{
			modelCallBudgetTestResponse("first", false),
			modelCallBudgetTestResponse("second", false),
		},
	}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(1200*time.Millisecond),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 900*time.Millisecond),
	)

	ch, err := wrapped.GenerateContent(ctx, modelCallBudgetTestRequest())
	require.NoError(t, err)
	require.Eventually(t, func() bool {
		return underlying.callCount() == 2
	}, 700*time.Millisecond, 10*time.Millisecond)

	first := <-ch
	require.Equal(t, "first", first.Choices[0].Message.Content)
	final := <-ch
	require.Equal(t, "final answer", final.Choices[0].Message.Content)
	_, ok := <-ch
	require.False(t, ok)
}

func TestModelCallBudgetModel_NilFinalChannelHonorsContext(t *testing.T) {
	t.Parallel()

	underlying := &scriptedPrefinalBudgetModel{nilFinal: true}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(500*time.Millisecond),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 300*time.Millisecond),
	)

	ch, err := wrapped.GenerateContent(ctx, modelCallBudgetTestRequest())
	require.NoError(t, err)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range ch {
		}
	}()
	select {
	case <-done:
	case <-time.After(800 * time.Millisecond):
		t.Fatal("nil final response channel ignored request context")
	}
	require.Equal(t, int64(2), underlying.callCount())
}

func TestModelCallBudgetIterModel_NilFinalChannelHonorsContext(
	t *testing.T,
) {
	t.Parallel()

	underlying := &scriptedPrefinalBudgetModel{nilFinal: true}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(500*time.Millisecond),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 300*time.Millisecond),
	)

	seq, err := iter.GenerateContentIter(
		ctx,
		modelCallBudgetTestRequest(),
	)
	require.NoError(t, err)
	done := make(chan struct{})
	go func() {
		defer close(done)
		seq(func(*model.Response) bool { return true })
	}()
	select {
	case <-done:
	case <-time.After(800 * time.Millisecond):
		t.Fatal("nil final response sequence ignored request context")
	}
	require.Equal(t, int64(2), underlying.callCount())
	require.Zero(t, underlying.iterCallCount())
}

func TestModelCallBudgetModel_FinalizesAfterInnerModelTimeout(
	t *testing.T,
) {
	t.Parallel()

	underlying := &innerTimeoutBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 150*time.Millisecond),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	ch, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)
	var got []*model.Response
	for resp := range ch {
		got = append(got, resp)
	}

	require.Len(t, got, 1)
	require.Equal(t, "final answer", got[0].Choices[0].Message.Content)
	requests := underlying.requestsSnapshot()
	require.Len(t, requests, 2)
	require.NotNil(t, requests[0].Tools)
	require.Nil(t, requests[1].Tools)
	require.Nil(t, req.Tools)
}

func TestModelCallBudgetIterModel_FinalizesAfterInnerModelTimeout(
	t *testing.T,
) {
	t.Parallel()

	underlying := &innerTimeoutBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, 150*time.Millisecond),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	seq, err := iter.GenerateContentIter(ctx, req)
	require.NoError(t, err)
	var got []*model.Response
	seq(func(resp *model.Response) bool {
		got = append(got, resp)
		return true
	})

	require.Len(t, got, 1)
	require.Equal(t, "final answer", got[0].Choices[0].Message.Content)
	requests := underlying.requestsSnapshot()
	require.Len(t, requests, 2)
	require.Empty(t, underlying.iterRequestsSnapshot())
	require.NotNil(t, requests[0].Tools)
	require.Nil(t, requests[1].Tools)
	require.Nil(t, req.Tools)
}

func TestModelCallBudgetModel_DoesNotFinalizeOutsideDeadlineWindow(
	t *testing.T,
) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Hour),
	)
	defer cancel()
	ctx = withModelCallBudgetValue(
		ctx,
		newModelCallBudget(0, false, time.Minute),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	_, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)

	got := underlying.lastRequest()
	require.NotNil(t, got)
	require.NotNil(t, got.Tools)
	require.Len(t, got.Messages, 1)
	require.NotNil(t, req.Tools)
}

func TestModelCallBudgetIterModel_FinalizesOnLastAllowedCall(t *testing.T) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx := withModelCallBudgetValue(
		context.Background(),
		newModelCallBudget(1, true, 0),
	)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
		ExtraFields: map[string]any{
			"parallel_tool_calls": true,
			"response_format":     "json",
			"tool_choice":         "required",
			"tools":               []string{"search"},
		},
	}

	_, err := iter.GenerateContentIter(ctx, req)
	require.NoError(t, err)

	got := underlying.lastIterRequest()
	require.NotNil(t, got)
	require.Nil(t, got.Tools)
	require.Len(t, got.Messages, 2)
	require.Contains(
		t,
		got.Messages[1].Content,
		"final allowed model call",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"Do not emit tool calls",
	)
	require.Contains(
		t,
		got.Messages[1].Content,
		"<tool_call>",
	)
	require.Nil(t, req.Tools)
	require.Len(t, req.Messages, 2)
	require.Equal(t, map[string]any{
		"response_format": "json",
	}, req.ExtraFields)
	require.Equal(t, map[string]any{
		"response_format": "json",
	}, got.ExtraFields)
}

func TestModelCallBudgetIterModel_CountFinalizationUsesNativeIter(
	t *testing.T,
) {
	t.Parallel()

	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	iter, ok := wrapped.(model.IterModel)
	require.True(t, ok)
	ctx := withModelCallBudgetValue(
		context.Background(),
		newModelCallBudget(1, true, time.Minute),
	)

	_, err := iter.GenerateContentIter(ctx, modelCallBudgetTestRequest())
	require.NoError(t, err)

	require.NotNil(t, underlying.lastIterRequest())
	require.Nil(t, underlying.lastRequest())
}

func TestApplyFinalModelCallRequestNil(t *testing.T) {
	t.Parallel()

	got := applyFinalModelCallRequest(nil)

	require.NotNil(t, got)
	require.Nil(t, got.Tools)
	require.Len(t, got.Messages, 1)
	require.Contains(t, got.Messages[0].Content, "final allowed model call")
}

func TestModelCallBudgetModel_UserPromptPrefixConsumesBudget(t *testing.T) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	ctx := withModelCallBudget(context.Background(), 1)
	req := &model.Request{Messages: []model.Message{model.NewUserMessage(
		"Analyze the following conversation between a user and an assistant, " +
			"and provide a concise summary.",
	)}}

	_, err := wrapped.GenerateContent(ctx, req)
	require.NoError(t, err)
	_, err = wrapped.GenerateContent(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 1, underlying.callCount())
}

func TestModelCallBudgetBypassModel_DoesNotConsumeContextBudget(
	t *testing.T,
) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	budgeted := newModelCallBudgetModel(underlying)
	bypassed := newModelCallBudgetBypassModel(budgeted)
	ctx := withModelCallBudget(context.Background(), 1)

	_, err := bypassed.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = bypassed.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)

	_, err = budgeted.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = budgeted.GenerateContent(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 3, underlying.callCount())
}

func TestModelCallBudgetBypassModel_DoesNotConsumeInvocationBudget(
	t *testing.T,
) {
	t.Parallel()

	underlying := &countingBudgetModel{}
	budgeted := newModelCallBudgetModel(underlying)
	bypassed := newModelCallBudgetBypassModel(budgeted)
	factory := newModelCallBudgetFactory(1, false, 0)
	inv := agent.NewInvocation(agent.WithInvocationRunOptions(
		agent.NewRunOptions(agent.MergeRuntimeState(map[string]any{
			modelCallBudgetRuntimeStateKey: factory,
		})),
	))
	ctx := agent.NewInvocationContext(context.Background(), inv)

	_, err := bypassed.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = bypassed.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)

	_, err = budgeted.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = budgeted.GenerateContent(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 3, underlying.callCount())
}

func TestModelCallBudgetBypassIterModel_DoesNotConsumeContextBudget(
	t *testing.T,
) {
	t.Parallel()

	underlying := &countingBudgetIterModel{}
	budgeted := newModelCallBudgetModel(underlying)
	bypassed := newModelCallBudgetBypassModel(budgeted)
	bypassIter, ok := bypassed.(model.IterModel)
	require.True(t, ok)
	budgetedIter, ok := budgeted.(model.IterModel)
	require.True(t, ok)
	ctx := withModelCallBudget(context.Background(), 1)

	_, err := bypassIter.GenerateContentIter(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = bypassIter.GenerateContentIter(ctx, &model.Request{})
	require.NoError(t, err)

	_, err = budgetedIter.GenerateContentIter(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = budgetedIter.GenerateContentIter(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 3, underlying.iterCallCount())
}

type countingBudgetModel struct {
	calls atomic.Int64
}

func (m *countingBudgetModel) GenerateContent(
	_ context.Context,
	_ *model.Request,
) (<-chan *model.Response, error) {
	m.calls.Add(1)
	ch := make(chan *model.Response, 1)
	ch <- &model.Response{Choices: []model.Choice{{
		Message: model.NewAssistantMessage("ok"),
	}}}
	close(ch)
	return ch, nil
}

func (m *countingBudgetModel) Info() model.Info {
	return model.Info{Name: "counting"}
}

func (m *countingBudgetModel) callCount() int64 {
	return m.calls.Load()
}

type capturingBudgetModel struct {
	mu       sync.Mutex
	last     *model.Request
	iterLast *model.Request
}

func (m *capturingBudgetModel) GenerateContent(
	_ context.Context,
	req *model.Request,
) (<-chan *model.Response, error) {
	m.mu.Lock()
	m.last = req
	m.mu.Unlock()
	ch := make(chan *model.Response, 1)
	ch <- &model.Response{}
	close(ch)
	return ch, nil
}

func (m *capturingBudgetModel) Info() model.Info {
	return model.Info{Name: "capturing"}
}

func (m *capturingBudgetModel) GenerateContentIter(
	_ context.Context,
	req *model.Request,
) (model.Seq[*model.Response], error) {
	m.mu.Lock()
	m.iterLast = req
	m.mu.Unlock()
	return func(yield func(*model.Response) bool) {
		yield(&model.Response{})
	}, nil
}

func (m *capturingBudgetModel) lastRequest() *model.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.last
}

func (m *capturingBudgetModel) lastIterRequest() *model.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.iterLast
}

type prefinalTimeoutBudgetModel struct {
	mu           sync.Mutex
	requests     []*model.Request
	iterRequests []*model.Request
}

type stalledPrefinalBudgetModel struct {
	calls     atomic.Int64
	iterCalls atomic.Int64
	blocked   chan struct{}
	once      sync.Once
}

type retainedBudgetRequest struct {
	hasTools bool
	messages int
	stream   bool
}

type retainingPrefinalBudgetModel struct {
	observed chan retainedBudgetRequest
}

func (m *retainingPrefinalBudgetModel) GenerateContent(
	ctx context.Context,
	req *model.Request,
) (<-chan *model.Response, error) {
	if req == nil || req.Tools == nil {
		ch := make(chan *model.Response, 1)
		ch <- modelCallBudgetTestFinalResponse()
		close(ch)
		return ch, nil
	}
	go func() {
		<-ctx.Done()
		m.observed <- retainedBudgetRequest{
			hasTools: req.Tools != nil,
			messages: len(req.Messages),
			stream:   req.Stream,
		}
	}()
	return make(chan *model.Response), nil
}

func (m *retainingPrefinalBudgetModel) Info() model.Info {
	return model.Info{Name: "retaining-prefinal"}
}

type scriptedPrefinalBudgetModel struct {
	calls                     atomic.Int64
	iterCalls                 atomic.Int64
	initial                   []*model.Response
	closeFirst                bool
	nilFinal                  bool
	finalCanceled             chan struct{}
	cancelOnce                sync.Once
	ctxMu                     sync.Mutex
	initialCtx                context.Context
	finalBeforePrefinalCancel atomic.Bool
}

func (m *scriptedPrefinalBudgetModel) GenerateContent(
	ctx context.Context,
	req *model.Request,
) (<-chan *model.Response, error) {
	m.calls.Add(1)
	if req == nil || req.Tools == nil {
		m.ctxMu.Lock()
		initialCtx := m.initialCtx
		m.ctxMu.Unlock()
		if initialCtx != nil && initialCtx.Err() == nil {
			m.finalBeforePrefinalCancel.Store(true)
		}
		if m.finalCanceled != nil {
			go func() {
				<-ctx.Done()
				m.cancelOnce.Do(func() { close(m.finalCanceled) })
			}()
		}
		if m.nilFinal {
			return nil, nil
		}
		ch := make(chan *model.Response, 1)
		ch <- modelCallBudgetTestFinalResponse()
		close(ch)
		return ch, nil
	}
	m.ctxMu.Lock()
	m.initialCtx = ctx
	m.ctxMu.Unlock()
	ch := make(chan *model.Response, len(m.initial))
	for _, resp := range m.initial {
		ch <- resp
	}
	if m.closeFirst {
		close(ch)
	}
	return ch, nil
}

func (m *scriptedPrefinalBudgetModel) Info() model.Info {
	return model.Info{Name: "scripted-prefinal"}
}

func (m *scriptedPrefinalBudgetModel) GenerateContentIter(
	_ context.Context,
	_ *model.Request,
) (model.Seq[*model.Response], error) {
	m.iterCalls.Add(1)
	return func(func(*model.Response) bool) {}, nil
}

func (m *scriptedPrefinalBudgetModel) callCount() int64 {
	return m.calls.Load()
}

func (m *scriptedPrefinalBudgetModel) iterCallCount() int64 {
	return m.iterCalls.Load()
}

func (m *scriptedPrefinalBudgetModel) finalStartedBeforeCancel() bool {
	return m.finalBeforePrefinalCancel.Load()
}

func newStalledPrefinalBudgetModel() *stalledPrefinalBudgetModel {
	return &stalledPrefinalBudgetModel{blocked: make(chan struct{})}
}

func (m *stalledPrefinalBudgetModel) GenerateContent(
	_ context.Context,
	req *model.Request,
) (<-chan *model.Response, error) {
	m.calls.Add(1)
	if req == nil || req.Tools == nil {
		ch := make(chan *model.Response, 1)
		ch <- modelCallBudgetTestFinalResponse()
		close(ch)
		return ch, nil
	}
	return make(chan *model.Response), nil
}

func (m *stalledPrefinalBudgetModel) Info() model.Info {
	return model.Info{Name: "stalled-prefinal"}
}

func (m *stalledPrefinalBudgetModel) GenerateContentIter(
	_ context.Context,
	req *model.Request,
) (model.Seq[*model.Response], error) {
	m.iterCalls.Add(1)
	return func(yield func(*model.Response) bool) {
		if req == nil || req.Tools == nil {
			yield(modelCallBudgetTestFinalResponse())
			return
		}
		<-m.blocked
	}, nil
}

func (m *stalledPrefinalBudgetModel) release() {
	m.once.Do(func() { close(m.blocked) })
}

func (m *stalledPrefinalBudgetModel) callCount() int64 {
	return m.calls.Load()
}

func (m *stalledPrefinalBudgetModel) iterCallCount() int64 {
	return m.iterCalls.Load()
}

func (m *prefinalTimeoutBudgetModel) GenerateContent(
	ctx context.Context,
	req *model.Request,
) (<-chan *model.Response, error) {
	m.mu.Lock()
	m.requests = append(m.requests, cloneBudgetTestRequest(req))
	m.mu.Unlock()
	ch := make(chan *model.Response, 1)
	go func() {
		defer close(ch)
		if req == nil || req.Tools == nil {
			ch <- modelCallBudgetTestFinalResponse()
			return
		}
		<-ctx.Done()
		ch <- &model.Response{
			Error: model.ResponseErrorFromError(
				ctx.Err(),
				model.ErrorTypeStreamError,
			),
			Done: true,
		}
	}()
	return ch, nil
}

func (m *prefinalTimeoutBudgetModel) Info() model.Info {
	return model.Info{Name: "prefinal-timeout"}
}

func (m *prefinalTimeoutBudgetModel) GenerateContentIter(
	ctx context.Context,
	req *model.Request,
) (model.Seq[*model.Response], error) {
	m.mu.Lock()
	m.iterRequests = append(m.iterRequests, cloneBudgetTestRequest(req))
	m.mu.Unlock()
	return func(yield func(*model.Response) bool) {
		if req == nil || req.Tools == nil {
			yield(modelCallBudgetTestFinalResponse())
			return
		}
		<-ctx.Done()
		yield(&model.Response{
			Error: model.ResponseErrorFromError(
				ctx.Err(),
				model.ErrorTypeStreamError,
			),
			Done: true,
		})
	}, nil
}

func (m *prefinalTimeoutBudgetModel) requestsSnapshot() []*model.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*model.Request(nil), m.requests...)
}

func (m *prefinalTimeoutBudgetModel) iterRequestsSnapshot() []*model.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*model.Request(nil), m.iterRequests...)
}

type innerTimeoutBudgetModel struct {
	mu           sync.Mutex
	requests     []*model.Request
	iterRequests []*model.Request
}

func (m *innerTimeoutBudgetModel) GenerateContent(
	_ context.Context,
	req *model.Request,
) (<-chan *model.Response, error) {
	m.mu.Lock()
	m.requests = append(m.requests, cloneBudgetTestRequest(req))
	m.mu.Unlock()
	ch := make(chan *model.Response, 1)
	if req == nil || req.Tools == nil {
		ch <- modelCallBudgetTestFinalResponse()
	} else {
		ch <- timeoutResponse(5*time.Minute, context.DeadlineExceeded)
	}
	close(ch)
	return ch, nil
}

func (m *innerTimeoutBudgetModel) Info() model.Info {
	return model.Info{Name: "inner-timeout"}
}

func (m *innerTimeoutBudgetModel) GenerateContentIter(
	_ context.Context,
	req *model.Request,
) (model.Seq[*model.Response], error) {
	m.mu.Lock()
	m.iterRequests = append(m.iterRequests, cloneBudgetTestRequest(req))
	m.mu.Unlock()
	return func(yield func(*model.Response) bool) {
		if req == nil || req.Tools == nil {
			yield(modelCallBudgetTestFinalResponse())
			return
		}
		yield(timeoutResponse(5*time.Minute, context.DeadlineExceeded))
	}, nil
}

func (m *innerTimeoutBudgetModel) requestsSnapshot() []*model.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*model.Request(nil), m.requests...)
}

func (m *innerTimeoutBudgetModel) iterRequestsSnapshot() []*model.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*model.Request(nil), m.iterRequests...)
}

func modelCallBudgetTestFinalResponse() *model.Response {
	return modelCallBudgetTestResponse("final answer", true)
}

func modelCallBudgetTestResponse(
	content string,
	done bool,
) *model.Response {
	return &model.Response{
		Done: done,
		Choices: []model.Choice{{
			Message: model.NewAssistantMessage(content),
		}},
	}
}

func modelCallBudgetTestRequest() *model.Request {
	return &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}
}

func cloneBudgetTestRequest(req *model.Request) *model.Request {
	if req == nil {
		return nil
	}
	clone := *req
	if req.Tools != nil {
		clone.Tools = make(map[string]tool.Tool, len(req.Tools))
		for name, t := range req.Tools {
			clone.Tools[name] = t
		}
	}
	clone.Messages = append([]model.Message(nil), req.Messages...)
	return &clone
}

type failingBudgetTokenCounter struct{}

func (failingBudgetTokenCounter) CountTokens(
	context.Context,
	model.Message,
) (int, error) {
	return 0, fmt.Errorf("count tokens")
}

func (failingBudgetTokenCounter) CountTokensRange(
	context.Context,
	[]model.Message,
	int,
	int,
) (int, error) {
	return 0, fmt.Errorf("count tokens range")
}

type countingBudgetIterModel struct {
	countingBudgetModel
	iterCalls atomic.Int64
}

func (m *countingBudgetIterModel) GenerateContentIter(
	_ context.Context,
	_ *model.Request,
) (model.Seq[*model.Response], error) {
	m.iterCalls.Add(1)
	return func(yield func(*model.Response) bool) {
		yield(&model.Response{})
	}, nil
}

func (m *countingBudgetIterModel) iterCallCount() int64 {
	return m.iterCalls.Load()
}

func TestBuildModelCallBudgetRunOptionResolverInjectsBudget(
	t *testing.T,
) {
	t.Parallel()

	resolver := buildModelCallBudgetRunOptionResolver(1, false, 0)
	ctx, runOpts, err := resolver(context.Background(), gateway.RunOptionInput{})
	require.NoError(t, err)
	require.Len(t, runOpts, 1)

	require.Nil(t, modelCallBudgetFromContext(ctx))

	opts := agent.NewRunOptions(runOpts...)
	rawFactory := opts.RuntimeState[modelCallBudgetRuntimeStateKey]
	factory, ok := rawFactory.(*modelCallBudgetFactory)
	require.True(t, ok)
	require.NotNil(t, factory)

	underlying := &countingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	parent := agent.NewInvocation(
		agent.WithInvocationID("parent"),
		agent.WithInvocationRunOptions(opts),
	)
	parentCtx := agent.NewInvocationContext(context.Background(), parent)
	child := parent.Clone(agent.WithInvocationID("child"))
	childCtx := agent.NewInvocationContext(context.Background(), child)

	_, err = wrapped.GenerateContent(parentCtx, &model.Request{})
	require.NoError(t, err)
	_, err = wrapped.GenerateContent(parentCtx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")

	_, err = wrapped.GenerateContent(childCtx, &model.Request{})
	require.NoError(t, err)
	_, err = wrapped.GenerateContent(childCtx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 2, underlying.callCount())
}

func TestBuildModelCallBudgetRunOptionResolverBypassesAuxiliaryCalls(
	t *testing.T,
) {
	t.Parallel()

	resolver := buildModelCallBudgetRunOptionResolver(1, false, 0)
	_, runOpts, err := resolver(context.Background(), gateway.RunOptionInput{})
	require.NoError(t, err)

	opts := agent.NewRunOptions(runOpts...)
	underlying := &countingBudgetModel{}
	budgeted := newModelCallBudgetModel(underlying)
	auxiliary := newModelCallBudgetBypassModel(budgeted)
	inv := agent.NewInvocation(
		agent.WithInvocationID("run"),
		agent.WithInvocationRunOptions(opts),
	)
	ctx := agent.NewInvocationContext(context.Background(), inv)

	_, err = auxiliary.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = auxiliary.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)

	_, err = budgeted.GenerateContent(ctx, &model.Request{})
	require.NoError(t, err)
	_, err = budgeted.GenerateContent(ctx, &model.Request{})
	require.ErrorContains(t, err, "max LLM calls (1) exceeded")
	require.EqualValues(t, 3, underlying.callCount())
}

func TestBuildModelCallBudgetRunOptionResolverInjectsDeadlineBudget(
	t *testing.T,
) {
	t.Parallel()

	resolver := buildModelCallBudgetRunOptionResolver(
		0,
		false,
		time.Minute,
	)
	ctx, runOpts, err := resolver(context.Background(), gateway.RunOptionInput{})
	require.NoError(t, err)
	require.Len(t, runOpts, 1)
	require.Nil(t, modelCallBudgetFromContext(ctx))

	opts := agent.NewRunOptions(runOpts...)
	underlying := &capturingBudgetModel{}
	wrapped := newModelCallBudgetModel(underlying)
	inv := agent.NewInvocation(
		agent.WithInvocationID("deadline-run"),
		agent.WithInvocationRunOptions(opts),
	)
	deadlineCtx, cancel := context.WithDeadline(
		context.Background(),
		time.Now().Add(time.Second),
	)
	defer cancel()
	deadlineCtx = agent.NewInvocationContext(deadlineCtx, inv)
	req := &model.Request{
		Messages: []model.Message{model.NewUserMessage("question")},
		Tools:    map[string]tool.Tool{"search": nil},
	}

	_, err = wrapped.GenerateContent(deadlineCtx, req)
	require.NoError(t, err)

	got := underlying.lastRequest()
	require.NotNil(t, got)
	require.Nil(t, got.Tools)
	require.Len(t, got.Messages, 2)
	require.Contains(
		t,
		got.Messages[1].Content,
		"final allowed model call",
	)
}

func TestModelCallBudgetFactoryReusesDeadlineBudgetForInvocation(
	t *testing.T,
) {
	t.Parallel()

	factory := newModelCallBudgetFactory(0, false, time.Minute)
	inv := agent.NewInvocation(agent.WithInvocationID("deadline-run"))

	first := factory.budgetFor(inv)
	second := factory.budgetFor(inv)

	require.NotNil(t, first)
	require.Same(t, first, second)
	require.Equal(t, time.Minute, first.deadlineWindow)
}
