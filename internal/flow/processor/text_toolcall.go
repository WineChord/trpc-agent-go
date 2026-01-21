package processor

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/google/uuid"
	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/event"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/tool"
)

const (
	toolCallTypeFunction = "function"
	finalAnswerTag       = "/*FINAL_ANSWER*/"
	toFunctionsPrefix    = "to=functions."
)

// TextToolCallResponseProcessor converts textual tool-call markers into
// structured tool calls, so the tool execution pipeline can run them.
//
// Some models may emit "To=functions.<tool>\n{...json...}" in plain text
// instead of producing structured tool_calls. When detected, this processor
// extracts the tool name and JSON arguments and populates ToolCalls.
type TextToolCallResponseProcessor struct{}

func NewTextToolCallResponseProcessor() *TextToolCallResponseProcessor {
	return &TextToolCallResponseProcessor{}
}

func (p *TextToolCallResponseProcessor) ProcessResponse(
	ctx context.Context,
	invocation *agent.Invocation,
	req *model.Request,
	rsp *model.Response,
	ch chan<- *event.Event,
) {
	_ = ctx
	_ = invocation
	_ = ch

	if rsp == nil || rsp.IsPartial || rsp.IsToolCallResponse() {
		return
	}
	if req == nil || len(req.Tools) == 0 {
		return
	}
	if len(rsp.Choices) == 0 {
		return
	}

	for i := range rsp.Choices {
		msg := &rsp.Choices[i].Message
		if msg.Content == "" {
			continue
		}
		if strings.Contains(msg.Content, finalAnswerTag) {
			continue
		}

		toolCall, cleaned, ok := parseTextToolCall(
			msg.Content,
			req.Tools,
		)
		if !ok {
			continue
		}

		msg.ToolCalls = []model.ToolCall{toolCall}
		msg.Content = cleaned
	}
}

func parseTextToolCall(
	content string,
	tools map[string]tool.Tool,
) (model.ToolCall, string, bool) {
	lower := strings.ToLower(content)
	idx := strings.Index(lower, toFunctionsPrefix)
	if idx == -1 {
		return model.ToolCall{}, content, false
	}

	nameStart := idx + len(toFunctionsPrefix)
	nameEnd := nameStart
	for nameEnd < len(content) && isToolNameChar(content[nameEnd]) {
		nameEnd++
	}
	if nameEnd == nameStart {
		return model.ToolCall{}, content, false
	}
	name := content[nameStart:nameEnd]
	if _, ok := tools[name]; !ok {
		return model.ToolCall{}, content, false
	}

	braceRel := strings.Index(content[nameEnd:], "{")
	if braceRel == -1 {
		return model.ToolCall{}, content, false
	}
	jsonStart := nameEnd + braceRel

	dec := json.NewDecoder(strings.NewReader(content[jsonStart:]))
	dec.UseNumber()
	var args any
	if err := dec.Decode(&args); err != nil {
		return model.ToolCall{}, content, false
	}

	argsMap, ok := args.(map[string]any)
	if !ok {
		return model.ToolCall{}, content, false
	}
	argsBytes, err := json.Marshal(argsMap)
	if err != nil {
		return model.ToolCall{}, content, false
	}

	callID := uuid.NewString()
	toolCall := model.ToolCall{
		Type: toolCallTypeFunction,
		ID:   callID,
		Function: model.FunctionDefinitionParam{
			Name:      name,
			Arguments: argsBytes,
		},
	}

	lineStart := strings.LastIndex(content[:idx], "\n")
	if lineStart == -1 {
		lineStart = 0
	} else {
		lineStart++
	}
	jsonEnd := jsonStart + int(dec.InputOffset())

	cleaned := content[:lineStart] + content[jsonEnd:]
	cleaned = strings.TrimSpace(cleaned)
	return toolCall, cleaned, true
}

func isToolNameChar(b byte) bool {
	switch {
	case b >= 'a' && b <= 'z':
		return true
	case b >= 'A' && b <= 'Z':
		return true
	case b >= '0' && b <= '9':
		return true
	case b == '_' || b == '-':
		return true
	default:
		return false
	}
}
