// `-w / --worktree`: create an isolated git worktree, chdir into it for the
// duration of the agent run, then optionally clean up on exit.
//
// Matches the hermes -w UX: the agent runs against a throwaway checkout of
// the current branch so parallel agents can't trample each other's working
// trees. Cleanup is best-effort — if the agent leaves changes behind we
// don't lose them.

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::process::Command;

pub struct Worktree {
    pub path: PathBuf,
    pub branch: String,
}

impl Worktree {
    pub fn create() -> Result<Self> {
        let head = Command::new("git").args(["rev-parse", "--abbrev-ref", "HEAD"]).output()?;
        if !head.status.success() {
            return Err(anyhow!("not in a git repository (or `git` not on PATH)"));
        }
        let base_branch = String::from_utf8_lossy(&head.stdout).trim().to_string();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let branch = format!("agent/{base_branch}/{nanos:x}");
        let path = std::env::temp_dir().join(format!("hermes-core-wt-{nanos:x}"));

        let out = Command::new("git")
            .args(["worktree", "add", "-b", &branch, path.to_str().unwrap(), &base_branch])
            .output()?;
        if !out.status.success() {
            return Err(anyhow!(
                "git worktree add failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(Self { path, branch })
    }

    pub fn cleanup(&self) -> Result<()> {
        let out = Command::new("git")
            .args(["worktree", "remove", "--force", self.path.to_str().unwrap()])
            .output()?;
        if !out.status.success() {
            return Err(anyhow!(
                "git worktree remove failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(())
    }
}
