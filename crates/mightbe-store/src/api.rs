//! mightbe-store 对外 API（README 4.4 / 11）。
//! 其他模块只允许 `use mightbe_store::api::*`。

use mightbe_core::MtbResult;

/// 存储引擎抽象。M0 仅有类型面；页引擎与加密在 M2 落地。
pub trait Store: Send + Sync {
    /// 打开/创建数据库（MDB 容器）
    fn open_database(&self, name: &str, opts: OpenOpts) -> MtbResult<DbHandle>;

    /// 表扫描（火山模型迭代器最小面）
    fn scan(&self, db: DbHandle, table: &str) -> MtbResult<RowIter>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DbHandle(u64);

impl DbHandle {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
    pub fn id(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct OpenOpts {
    /// false 时拒绝创建不存在的库（MySQL EXISTS 语义）
    pub create_if_missing: bool,
}

impl Default for OpenOpts {
    fn default() -> Self {
        Self { create_if_missing: true }
    }
}

/// 一行数据：有序槽位，类型系统在 sql 层。
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub slots: Vec<Slot>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Slot {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Text(String),
    Bytes(Vec<u8>),
    /// VECTOR(n) / VECTOR（README 8.10/9）
    Vector(Vec<f32>),
}

/// 列定义。`col_type` 用 [`crate::table::col`] 里的标签常量，与行内编码同源。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub col_type: u8,
    pub primary_key: bool,
}

/// 表定义。Catalog 里的一条记录，也是建表/删表的唯一事实来源。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDef {
    pub name: String,
    /// 表堆的根页号（页链首）
    pub root_page: u64,
    /// 下一个可用的行号（表内自增，从 1 起）
    pub next_rid: u64,
    pub columns: Vec<ColumnDef>,
}

impl TableDef {
    /// 按列名取列序；找不到返回 `None`。
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// 主键列名；没有主键时返回 `None`。
    pub fn primary_key(&self) -> Option<&str> {
        self.columns
            .iter()
            .find(|c| c.primary_key)
            .map(|c| c.name.as_str())
    }
}

/// 简化行迭代器（M2 换成真正的火山模型迭代器，支持谓词下推）
pub struct RowIter {
    rows: std::vec::IntoIter<Row>,
}

impl RowIter {
    pub fn new(rows: Vec<Row>) -> Self {
        Self { rows: rows.into_iter() }
    }
}

impl Iterator for RowIter {
    type Item = Row;
    fn next(&mut self) -> Option<Self::Item> {
        self.rows.next()
    }
}
