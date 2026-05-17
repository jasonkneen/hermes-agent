use anyhow::Result;
use std::path::{Path, PathBuf};
use tokio::io::AsyncBufReadExt;

use crate::types::Message;

pub fn session_dir(id: &str) -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hermes-core")
        .join("sessions")
        .join(id)
}

pub fn transcript_path(id: &str) -> PathBuf {
    session_dir(id).join("messages.jsonl")
}

pub async fn load(path: &Path) -> Result<Vec<Message>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = tokio::fs::File::open(path).await?;
    let mut reader = tokio::io::BufReader::new(file).lines();
    let mut out = Vec::new();
    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Message>(&line) {
            Ok(m) => out.push(m),
            Err(e) => eprintln!("skipping malformed transcript line: {e}"),
        }
    }
    Ok(out)
}

pub async fn append(path: &Path, msg: &Message) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    let line = serde_json::to_string(msg)?;
    let mut f = tokio::fs::OpenOptions::new().create(true).append(true).open(path).await?;
    use tokio::io::AsyncWriteExt;
    f.write_all(line.as_bytes()).await?;
    f.write_all(b"\n").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Block;
    use serde_json::json;

    #[tokio::test]
    async fn jsonl_round_trip() {
        let dir = tempdir();
        let path = dir.join("messages.jsonl");
        let msgs = vec![
            Message::User { content: vec![Block::Text { text: "hello".into() }] },
            Message::Assistant {
                content: vec![Block::ToolUse {
                    id: "abc".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                }],
            },
            Message::User {
                content: vec![Block::ToolResult {
                    tool_use_id: "abc".into(),
                    content: "file.txt".into(),
                    is_error: false,
                }],
            },
        ];
        for m in &msgs {
            append(&path, m).await.unwrap();
        }
        let loaded = load(&path).await.unwrap();
        assert_eq!(loaded.len(), 3);
        match &loaded[0] {
            Message::User { content } => match &content[0] {
                Block::Text { text } => assert_eq!(text, "hello"),
                _ => panic!(),
            },
            _ => panic!(),
        }
        match &loaded[1] {
            Message::Assistant { content } => match &content[0] {
                Block::ToolUse { id, name, .. } => {
                    assert_eq!(id, "abc");
                    assert_eq!(name, "bash");
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn load_missing_returns_empty() {
        let dir = tempdir();
        let loaded = load(&dir.join("nope.jsonl")).await.unwrap();
        assert!(loaded.is_empty());
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("hermes-core-test-{}", std::process::id()))
            .join(uuid_like());
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn uuid_like() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        format!("{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos())
    }
}
