package processor

import (
	"context"
	"encoding/json"
	"testing"

	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/tool"
)

func TestParseTextToolCall(t *testing.T) {
	tools := map[string]tool.Tool{
		"web_search": nil,
	}

	content := "/*ACTION*/\nTo=functions.web_search\n" +
		"{\"query\":\"hello\",\"max_results\":2}\n"

	tc, cleaned, ok := parseTextToolCall(content, tools)
	if !ok {
		t.Fatal("parseTextToolCall() ok=false")
	}
	if tc.Type != toolCallTypeFunction {
		t.Fatalf("Type=%q want %q", tc.Type, toolCallTypeFunction)
	}
	if tc.ID == "" {
		t.Fatal("ID is empty")
	}
	if tc.Function.Name != "web_search" {
		t.Fatalf("Name=%q want %q", tc.Function.Name, "web_search")
	}
	if len(tc.Function.Arguments) == 0 {
		t.Fatal("Arguments is empty")
	}

	var gotArgs map[string]any
	if err := json.Unmarshal(tc.Function.Arguments, &gotArgs); err != nil {
		t.Fatalf("Unmarshal args: %v", err)
	}
	if gotArgs["query"] != "hello" {
		t.Fatalf("query=%v want %q", gotArgs["query"], "hello")
	}
	if gotArgs["max_results"] != float64(2) {
		t.Fatalf("max_results=%v want %v", gotArgs["max_results"], 2)
	}

	if cleaned != "/*ACTION*/" {
		t.Fatalf("cleaned=%q want %q", cleaned, "/*ACTION*/")
	}
}

func TestParseTextToolCall_UnknownTool(t *testing.T) {
	tools := map[string]tool.Tool{
		"web_fetch": nil,
	}
	content := "To=functions.web_search\n{\"query\":\"hello\"}"

	_, _, ok := parseTextToolCall(content, tools)
	if ok {
		t.Fatal("parseTextToolCall() ok=true want false")
	}
}

func TestTextToolCallRespProc_SkipFinalAnswer(t *testing.T) {
	p := NewTextToolCallResponseProcessor()
	req := &model.Request{
		Tools: map[string]tool.Tool{
			"web_search": nil,
		},
	}
	rsp := &model.Response{
		Choices: []model.Choice{
			{
				Message: model.Message{
					Role: model.RoleAssistant,
					Content: finalAnswerTag + "\n" +
						"To=functions.web_search\n" +
						"{\"query\":\"hello\"}",
				},
			},
		},
	}

	p.ProcessResponse(context.Background(), &agent.Invocation{}, req, rsp, nil)

	if rsp.IsToolCallResponse() {
		t.Fatal("expected no tool calls")
	}
}

func TestTextToolCallRespProc_SkipWhenToolCallsPresent(t *testing.T) {
	p := NewTextToolCallResponseProcessor()
	req := &model.Request{
		Tools: map[string]tool.Tool{
			"web_search": nil,
		},
	}
	rsp := &model.Response{
		Choices: []model.Choice{
			{
				Message: model.Message{
					Role: model.RoleAssistant,
					Content: "To=functions.web_search\n" +
						"{\"query\":\"hello\"}",
					ToolCalls: []model.ToolCall{
						{
							Type: toolCallTypeFunction,
							ID:   "existing",
							Function: model.FunctionDefinitionParam{
								Name: "web_search",
							},
						},
					},
				},
			},
		},
	}

	p.ProcessResponse(context.Background(), &agent.Invocation{}, req, rsp, nil)

	if len(rsp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatal("expected existing tool calls preserved")
	}
	if rsp.Choices[0].Message.ToolCalls[0].ID != "existing" {
		t.Fatal("expected existing tool call preserved")
	}
}
