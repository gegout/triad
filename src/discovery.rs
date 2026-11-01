use anyhow::Result;
use crate::execution::{AntigravityAdapter, BubblewrapAdapter, ExecutionAdapter};
use crate::model::ModelClient;
use crate::store::Store;
use crate::web::WebExplorer;

pub struct DiscoveryReport {
    pub os_info: String,
    pub rust_info: String,
    pub bubblewrap_info: String,
    pub antigravity_info: String,
    pub chat_snap_info: String,
    pub embedding_snap_info: String,
    pub network_info: String,
    pub storage_info: String,
}

pub struct DiscoveryService {
    store: Store,
    model_client: ModelClient,
    web_explorer: WebExplorer,
    workspace_dir: std::path::PathBuf,
}

impl DiscoveryService {
    pub fn new(
        store: Store,
        model_client: ModelClient,
        web_explorer: WebExplorer,
        workspace_dir: std::path::PathBuf,
    ) -> Self {
        Self {
            store,
            model_client,
            web_explorer,
            workspace_dir,
        }
    }

    pub async fn run_probes(&self) -> Result<DiscoveryReport> {
        // 1. Operating system & architecture
        let os_desc = format!("Linux {} (x86_64)", std::env::consts::OS);
        self.store.record_environment_observation("os", &os_desc, "observed")?;

        // 2. Rust toolchain
        let rust_output = std::process::Command::new("rustc").arg("--version").output();
        let rust_info = match rust_output {
            Ok(out) if out.status.success() => {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                self.store.record_environment_observation("toolchain", &s, "observed")?;
                s
            }
            _ => {
                let s = "rustc unavailable".to_string();
                self.store.record_environment_observation("toolchain", &s, "missing")?;
                s
            }
        };

        // 3. Bubblewrap isolation
        let bwrap = BubblewrapAdapter::probe();
        let bwrap_info = bwrap.status_summary();
        let bwrap_status = if bwrap.is_available() { "observed" } else { "missing" };
        self.store.record_environment_observation("isolation", &bwrap_info, bwrap_status)?;

        // 4. Antigravity runtime
        let agy = AntigravityAdapter::probe("");
        let agy_info = agy.status_summary();
        let agy_status = if agy.is_available() { "observed" } else { "missing" };
        self.store.record_environment_observation("antigravity", &agy_info, agy_status)?;

        // 5. Chat snap endpoint
        let chat_url = "http://localhost:8366/v1";
        let chat_models = self.model_client.probe_models(chat_url).await;
        let chat_ok = chat_models.is_ok();
        let chat_snap_info = match chat_models {
            Ok(models) => {
                let info = format!("Connected to SmolLM3 snap; Advertised: {}", models.join(", "));
                self.store.record_environment_observation("models.chat", &info, "observed")?;
                info
            }
            Err(e) => {
                let info = format!("Chat snap probe failed: {}", e);
                self.store.record_environment_observation("models.chat", &info, "unavailable")?;
                info
            }
        };

        // 6. Embedding snap endpoint
        let emb_url = "http://127.0.0.1:8348/v1";
        let emb_models = self.model_client.probe_models(emb_url).await;
        let emb_ok = emb_models.is_ok();
        let embedding_snap_info = match emb_models {
            Ok(models) => {
                let info = format!("Connected to Nomic embedding snap; Advertised: {}", models.join(", "));
                self.store.record_environment_observation("models.embedding", &info, "observed")?;
                info
            }
            Err(e) => {
                let info = format!("Embedding snap probe failed: {}", e);
                self.store.record_environment_observation("models.embedding", &info, "unavailable")?;
                info
            }
        };

        // 7. Network connectivity
        let net_ok = self.web_explorer.test_connectivity().await;
        let net_info = if net_ok {
            "Internet connectivity verified (Wikipedia reachable)".to_string()
        } else {
            "Internet connectivity offline or unreachable".to_string()
        };
        let net_status = if net_ok { "observed" } else { "unavailable" };
        self.store.record_environment_observation("network", &net_info, net_status)?;

        // 8. Storage & workspace
        let storage_info = format!("Workspace ready at {}", self.workspace_dir.display());
        self.store.record_environment_observation("storage", &storage_info, "observed")?;

        // 9. Initialise or update Capability Map
        self.init_or_update_capability_map(&bwrap, &agy, chat_ok, emb_ok, net_ok)?;

        Ok(DiscoveryReport {
            os_info: os_desc,
            rust_info,
            bubblewrap_info: bwrap_info,
            antigravity_info: agy_info,
            chat_snap_info,
            embedding_snap_info,
            network_info: net_info,
            storage_info,
        })
    }

    fn init_or_update_capability_map(
        &self,
        bwrap: &BubblewrapAdapter,
        agy: &AntigravityAdapter,
        chat_ok: bool,
        emb_ok: bool,
        net_ok: bool,
    ) -> Result<()> {
        let bwrap_status = if bwrap.is_available() { "demonstrated" } else { "missing" };
        self.store.upsert_capability_map(
            "isolated_tool_execution",
            bwrap_status,
            "Bubblewrap container sandboxing for compiling and executing tools without host risk",
            Some("Safe synthesis of new tools and experimental runners"),
        )?;

        let agy_status = if agy.is_available() { "demonstrated" } else { "missing" };
        self.store.upsert_capability_map(
            "antigravity_adapter",
            agy_status,
            "Antigravity SDK CLI adapter for host-verified process execution",
            Some("Multi-agent task distribution and automated pipeline execution"),
        )?;

        let reasoning_status = if chat_ok { "demonstrated" } else { "missing" };
        self.store.upsert_capability_map(
            "local_reasoning_chat",
            reasoning_status,
            "SmolLM3 local chat inference for hypothesis formulation and critical dialogue",
            Some("Autonomous exploration, experiment design, and debate"),
        )?;

        let emb_status = if emb_ok { "demonstrated" } else { "missing" };
        self.store.upsert_capability_map(
            "semantic_vector_retrieval",
            emb_status,
            "Nomic embedding inference with cosine search across memory claims and sources",
            Some("Knowledge retrieval and memory reuse across cycles"),
        )?;

        let net_status = if net_ok { "demonstrated" } else { "missing" };
        self.store.upsert_capability_map(
            "public_source_acquisition",
            net_status,
            "HTTP fetching of Wikipedia, arXiv, and open tech sources",
            Some("Discovery of contemporary research papers and data beyond local storage"),
        )?;

        self.store.upsert_capability_map(
            "dynamic_tool_synthesis",
            if bwrap.is_available() { "demonstrated" } else { "struggles" },
            "Automated generation of Rust tool code, sandboxed test validation, and binary registration",
            Some("Self-directed capability expansion for specialized text and data processing"),
        )?;

        self.store.upsert_capability_map(
            "weight_fine_tuning",
            "missing",
            "On-device gradient update / backpropagation of local models",
            Some("Model weights remain invariant; Triad improves through tools, memory, retrieval, and prompt coordination"),
        )?;

        Ok(())
    }
}
