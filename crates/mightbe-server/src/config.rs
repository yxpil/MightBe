//! `config/mightbe.toml` 的解析与默认值兜底（README 第 13 章）。
//!
//! 这里是全仓 **唯一** 允许出现"结构性默认值"的地方：超参、词表、网络结构仍归
//! 各自的 TOML，本模块只承载运行配置。

use mightbe_core::{MtbError, MtbResult};
use serde::Deserialize;

/// 顶层配置。缺省字段取各节 [`Default`]，因此空文件等价于全默认。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MightBeConfig {
    pub server: ServerCfg,
    pub storage: StorageCfg,
    pub engine: EngineCfg,
    pub nlp: NlpCfg,
    pub plugins: PluginsCfg,
    pub schema_learning: SchemaLearningCfg,
}

impl MightBeConfig {
    pub const DEFAULT_PATH: &'static str = "config/mightbe.toml";

    /// 读 TOML 配置文件；文件不存在时返回全默认配置。
    pub fn load(path: &str) -> MtbResult<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml(&text)
                .map_err(|e| MtbError::coded(MtbError::CONFIG_INVALID, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn from_toml(text: &str) -> Result<Self, String> {
        toml::from_str(text).map_err(|e| format!("{}: {}", MtbError::CONFIG_INVALID, e))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerCfg {
    pub host: String,
    pub port: u16,
}

impl Default for ServerCfg {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 9527,
        }
    }
}

impl ServerCfg {
    pub fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StorageCfg {
    pub data_dir: String,
    pub page_size: usize,
    pub wal_flush_ms: u64,
    /// `aes-256-gcm`（M2 唯一实现）；其余取值保留给远期
    pub mdb_cipher: String,
    /// 口令来源环境变量；为空则由启动方交互输入
    pub password_env: String,
}

impl Default for StorageCfg {
    fn default() -> Self {
        Self {
            data_dir: "data".into(),
            page_size: 8192,
            wal_flush_ms: 50,
            mdb_cipher: "aes-256-gcm".into(),
            password_env: String::new(),
        }
    }
}

impl StorageCfg {
    /// 页大小必须是 2 的幂且不小于页头 + 槽目录的可用下界。
    pub fn validate(&self) -> Result<(), String> {
        if self.page_size < 512 || !self.page_size.is_power_of_two() {
            return Err(format!("page_size 必须是 ≥512 的 2 的幂，当前 {}", self.page_size));
        }
        if self.mdb_cipher != "aes-256-gcm" {
            return Err(format!("mdb_cipher 暂只支持 aes-256-gcm，当前 {}", self.mdb_cipher));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct EngineCfg {
    /// naive | simd | cuda | opencl | npu | auto
    pub backend: String,
    /// auto | cpu | cuda:0 | npu:0
    pub device: String,
    /// 0 = 按核数
    pub threads: usize,
    pub profile: bool,
    pub gpu_memory_pool_mb: usize,
}

impl Default for EngineCfg {
    fn default() -> Self {
        Self {
            backend: "naive".into(),
            device: "auto".into(),
            threads: 0,
            profile: false,
            gpu_memory_pool_mb: 2048,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NlpCfg {
    pub stopwords: Vec<String>,
    /// tfidf | textrank
    pub keyword_algo: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PluginsCfg {
    pub dir: String,
    pub allow_reload: bool,
}

impl Default for PluginsCfg {
    fn default() -> Self {
        Self {
            dir: "plugins".into(),
            allow_reload: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SchemaLearningCfg {
    pub enabled: bool,
    pub auto_profile_on_write: bool,
    pub auto_learn_on_write: bool,
    pub relation_network: String,
    pub min_support: u64,
    pub sample_rows: u64,
    pub candidate_blockers: Vec<String>,
    /// REASON 默认搜索域：WORD | FIELD | TABLE
    pub default_scope: Vec<String>,
}

impl Default for SchemaLearningCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_profile_on_write: true,
            auto_learn_on_write: false,
            relation_network: String::new(),
            min_support: 50,
            sample_rows: 100_000,
            candidate_blockers: vec![
                "name_ngram".into(),
                "same_type".into(),
                "value_overlap".into(),
                "token_overlap".into(),
            ],
            default_scope: vec!["WORD".into()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_yields_defaults() {
        let cfg = MightBeConfig::from_toml("").unwrap();
        assert_eq!(cfg.server.port, 9527);
        assert_eq!(cfg.storage.page_size, 8192);
        assert_eq!(cfg.schema_learning.min_support, 50);
    }

    #[test]
    fn readme_sample_parses() {
        let text = r#"
[server]
host = "127.0.0.1"
port = 9527

[storage]
data_dir    = "data"
page_size   = 8192
wal_flush_ms = 50
mdb_cipher  = "aes-256-gcm"
password_env = "MIGHTBE_KEY"

[engine]
backend     = "naive"
device      = "auto"
threads     = 0
profile     = false
gpu_memory_pool_mb = 2048

[nlp]
stopwords = ["config/nlp/stopwords.zh.txt", "config/nlp/stopwords.en.txt"]
keyword_algo = "tfidf"

[plugins]
dir = "plugins"
allow_reload = true

[schema_learning]
enabled = true
auto_profile_on_write = true
auto_learn_on_write = false
relation_network = "config/networks/_relation_net.toml"
min_support = 50
sample_rows = 100000
candidate_blockers = ["name_ngram", "same_type", "value_overlap", "token_overlap"]
default_scope = ["WORD"]
"#;
        let cfg = MightBeConfig::from_toml(text).unwrap();
        assert_eq!(cfg.server.addr(), "127.0.0.1:9527");
        assert_eq!(cfg.nlp.stopwords.len(), 2);
        assert_eq!(cfg.storage.password_env, "MIGHTBE_KEY");
        cfg.storage.validate().unwrap();
    }

    #[test]
    fn unknown_key_is_rejected() {
        let err = MightBeConfig::from_toml("[server]\nportx = 1\n").unwrap_err();
        assert!(err.contains("unknown field"));
    }

    #[test]
    fn bad_page_size_rejected() {
        let cfg = StorageCfg {
            page_size: 4000,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn missing_file_falls_back_to_defaults() {
        let cfg = MightBeConfig::load("no/such/mightbe.toml").unwrap();
        assert_eq!(cfg.engine.backend, "naive");
    }
}
