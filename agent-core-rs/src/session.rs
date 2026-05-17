use anyhow::Result;
use std::path::{Path, PathBuf};
use tokio::io::AsyncBufReadExt;

use crate::types::Message;

pub fn session_dir(id: &str) -> PathBuf {
    let base = dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hermes-core")
        .join("sessions")
        .join(id);
    base
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
