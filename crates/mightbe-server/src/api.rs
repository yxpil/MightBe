//! mightbe-server 对外 API（README 4.6）。
//!
//! server 位于依赖链顶端，不被任何内部 crate 依赖，因此这里只放
//! 服务生命周期类型与线协议响应面，供 bin 与集成测试使用。

use mightbe_core::MtbResult;

/// 服务生命周期：启动装配 → 监听 → 优雅停机。
pub trait Service: Send + Sync {
    /// 完成装配并开始监听；返回实际绑定地址 `host:port`。
    fn start(&self) -> MtbResult<String>;

    /// 停止接收新连接，等待在途请求与训练步落检查点后返回。
    fn shutdown(&self) -> MtbResult<()>;
}

/// 一次已解析的客户端请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// 一条以 `;` 结尾的 SQL 语句
    Statement(String),
    /// 元命令（`.timer` / `.db` 等，cli 侧展开，server 侧忽略）
    Meta(String),
}

/// 线协议响应：`OK <cols>` + 数据行 + `END (rows=N, ms=X)`，错误为 `ERR <code> <msg>`。
#[derive(Debug, Clone)]
pub enum Response {
    Rows {
        cols: Vec<String>,
        rows: Vec<Vec<String>>,
        elapsed_ms: u128,
    },
    Affected {
        n: u64,
        info: String,
    },
    Job {
        job_id: String,
        info: String,
    },
    Empty {
        info: String,
    },
    Error {
        code: u16,
        message: String,
    },
}

impl Response {
    /// 渲染为线协议文本（不含结尾换行）。
    pub fn to_wire(&self) -> String {
        match self {
            Response::Rows {
                cols,
                rows,
                elapsed_ms,
            } => {
                let mut out = String::with_capacity(64 + cols.len() * 16 + rows.len() * 32);
                out.push_str("OK ");
                out.push_str(&cols.join("|"));
                for row in rows {
                    out.push('\n');
                    out.push_str(&row.join("|"));
                }
                out.push_str(&format!("\nEND (rows={}, ms={})", rows.len(), elapsed_ms));
                out
            }
            Response::Affected { n, info } => {
                format!("OK affected={}\nINFO {}\nEND (rows=1, ms=0)", n, info)
            }
            Response::Job { job_id, info } => {
                format!("OK job={}\nINFO {}\nEND (rows=0, ms=0)", job_id, info)
            }
            Response::Empty { info } => format!("OK\nINFO {}\nEND (rows=0, ms=0)", info),
            Response::Error { code, message } => format!("ERR {} {}", code, message),
        }
    }
}
