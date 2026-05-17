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
