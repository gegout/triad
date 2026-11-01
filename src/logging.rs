use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    pub timestamp: String,
    pub level: String,
    pub event: String,
    pub agent: Option<String>,
    pub job_id: Option<String>,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct Logger {
    log_dir: PathBuf,
    rotate_size_bytes: u64,
    retained_files: usize,
    lock: Arc<Mutex<()>>,
}

impl Logger {
    pub fn new(log_dir: PathBuf, rotate_size_mb: usize, retained_files: usize) -> Self {
        let _ = fs::create_dir_all(&log_dir);
        Self {
            log_dir,
            rotate_size_bytes: (rotate_size_mb as u64) * 1024 * 1024,
            retained_files,
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn log(
        &self,
        level: &str,
        event: &str,
        agent: Option<&str>,
        job_id: Option<&str>,
        message: &str,
        data: Option<serde_json::Value>,
    ) {
        let _guard = self.lock.lock().unwrap();
        let record = LogRecord {
            timestamp: Utc::now().to_rfc3339(),
            level: level.to_string(),
            event: event.to_string(),
            agent: agent.map(ToString::to_string),
            job_id: job_id.map(ToString::to_string),
            message: message.to_string(),
            data,
        };

        let current_log = self.log_dir.join("triad.jsonl");

        // Rotate if needed
        if let Ok(meta) = fs::metadata(&current_log) {
            if meta.len() >= self.rotate_size_bytes && self.rotate_size_bytes > 0 {
                self.rotate_logs(&current_log);
            }
        }

        if let Ok(line) = serde_json::to_string(&record) {
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&current_log) {
                let _ = writeln!(f, "{}", line);
            }
        }
    }

    fn rotate_logs(&self, current: &Path) {
        for i in (1..self.retained_files).rev() {
            let from = self.log_dir.join(format!("triad.{}.jsonl", i));
            let to = self.log_dir.join(format!("triad.{}.jsonl", i + 1));
            if from.exists() {
                let _ = fs::rename(from, to);
            }
        }
        let first_rot = self.log_dir.join("triad.1.jsonl");
        let _ = fs::rename(current, first_rot);
    }

    pub fn read_recent_records(&self, limit: usize, filter_agent: Option<&str>) -> Vec<LogRecord> {
        let current_log = self.log_dir.join("triad.jsonl");
        if !current_log.exists() {
            return Vec::new();
        }

        let content = fs::read_to_string(current_log).unwrap_or_default();
        let mut records = Vec::new();

        for line in content.lines().rev() {
            if let Ok(rec) = serde_json::from_str::<LogRecord>(line) {
                if let Some(agent) = filter_agent {
                    if rec.agent.as_deref() != Some(agent) {
                        continue;
                    }
                }
                records.push(rec);
                if records.len() >= limit {
                    break;
                }
            }
        }
        records.reverse();
        records
    }
}
