use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User { content: Vec<Block> },
    Assistant { content: Vec<Block> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: String, #[serde(default)] is_error: bool },
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Default)]
pub struct AssistantTurn {
    pub blocks: Vec<Block>,
    #[allow(dead_code)]
    pub stop_reason: String,
}

impl AssistantTurn {
    pub fn has_tool_calls(&self) -> bool {
        self.blocks.iter().any(|b| matches!(b, Block::ToolUse { .. }))
    }
    pub fn final_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| if let Block::Text { text } = b { Some(text.as_str()) } else { None })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn block_wire_tags_match_anthropic_spec() {
        let t = Block::Text { text: "hi".into() };
        let v = serde_json::to_value(&t).unwrap();
        assert_eq!(v["type"], "text");
        assert_eq!(v["text"], "hi");

        let tu = Block::ToolUse {
            id: "id1".into(),
            name: "bash".into(),
            input: json!({"command": "ls"}),
        };
        let v = serde_json::to_value(&tu).unwrap();
        assert_eq!(v["type"], "tool_use");
        assert_eq!(v["id"], "id1");
        assert_eq!(v["input"]["command"], "ls");

        let tr = Block::ToolResult {
            tool_use_id: "id1".into(),
            content: "ok".into(),
            is_error: false,
        };
        let v = serde_json::to_value(&tr).unwrap();
        assert_eq!(v["type"], "tool_result");
        assert_eq!(v["tool_use_id"], "id1");
        assert_eq!(v["content"], "ok");
        assert_eq!(v["is_error"], false);

        let th = Block::Thinking { thinking: "reasoning".into() };
        let v = serde_json::to_value(&th).unwrap();
        assert_eq!(v["type"], "thinking");
        assert_eq!(v["thinking"], "reasoning");
    }

    #[test]
    fn message_round_trips() {
        let m = Message::User {
            content: vec![
                Block::Text { text: "hi".into() },
                Block::ToolResult { tool_use_id: "x".into(), content: "y".into(), is_error: true },
            ],
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&s).unwrap();
        match back {
            Message::User { content } => assert_eq!(content.len(), 2),
            _ => panic!(),
        }
    }

    #[test]
    fn assistant_turn_helpers() {
        let t = AssistantTurn {
            blocks: vec![
                Block::Text { text: "a".into() },
                Block::ToolUse { id: "i".into(), name: "n".into(), input: json!({}) },
                Block::Text { text: "b".into() },
            ],
            stop_reason: "".into(),
        };
        assert!(t.has_tool_calls());
        assert_eq!(t.final_text(), "ab");
    }
}
