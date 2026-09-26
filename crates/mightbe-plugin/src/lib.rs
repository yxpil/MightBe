//! # mightbe-plugin
//! 自定义层插件 SDK（README 4.1.1 路径 C）。
//!
//! 用户实现 [`MbLayer`]，用 [`mb_export!`] 导出 C ABI，编译为 `crate-type = ["cdylib"]`，
//! 由框架 `CREATE LIBRARY ... FROM '<dll>'` 加载。
//!
//! 本 crate 是**叶子**：零依赖，宿主（`mightbe-core::plugin`）复用这里的 ABI 布局类型，
//! 保证两侧的结构体定义不会各自漂移。
//!
//! ABI 设计约束（与 README 4.1.1 一致）：
//! - 跨边界只传 [`HostTensor`]（定长 dims + 裸指针，只读视图）与 [`HostBuf`]
//!   （**宿主**分配、宿主回收的缓冲）；插件产出由 SDK 复制进宿主内存，插件自身内存
//!   永不跨越边界，因此两侧静态链接各自的分配器也安全；
//! - 插件**不持有权重**：`mb_param` 只声明参数名与形状，分配/初始化/梯度累积都在宿主侧；
//! - 每个导出函数内部先 `catch_unwind`，插件 panic 转成错误码 + [`mb_last_error`]，
//!   绝不跨越 C ABI 展开（Rust 1.81 起跨 `extern "C"` 展开会直接 abort）。

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};

/// 插件 ABI 版本。宿主拒绝版本不符的动态库（错误码 3101）。
pub const MB_ABI_VERSION: u32 = 1;

/// 跨 ABI 张量的最大秩，与 `mightbe-core::autograd::MAX_PERM_RANK` 对齐。
pub const MAX_RANK: usize = 4;

/// 导出符号表。宿主逐个 `GetProcAddress`/`dlsym`，缺任何一个报 3102。
pub const MB_SYMBOLS: [&str; 10] = [
    "mb_api_version",
    "mb_layer_count",
    "mb_describe",
    "mb_create",
    "mb_param",
    "mb_infer_shapes",
    "mb_forward",
    "mb_backward",
    "mb_drop",
    "mb_last_error",
];

// ───────────────────────── 跨 ABI 数据布局 ─────────────────────────

/// 只读张量视图：宿主 → 插件。`ptr` 为空表示该槽位无数据。
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct HostTensor {
    pub ptr: *const f32,
    pub len: u64,
    pub ndim: u32,
    pub dims: [u32; MAX_RANK],
}

/// 可写缓冲：由 `HostApi::alloc` 分配、宿主负责回收。插件用 SDK 填好后回传。
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct HostBuf {
    pub ptr: *mut f32,
    pub len: u64,
    pub ndim: u32,
    pub dims: [u32; MAX_RANK],
}

impl HostBuf {
    /// 空缓冲（"本槽位没有梯度"）。
    pub const NIL: HostBuf = HostBuf {
        ptr: std::ptr::null_mut(),
        len: 0,
        ndim: 0,
        dims: [0; MAX_RANK],
    };

    /// 宿主侧：把插件回传的缓冲重新收回到 `Vec`（所有权归调用方，只可收回一次）。
    ///
    /// # Safety
    /// 内存必须由本宿主的 [`HostApi::alloc`] 配出，且 `self` 之后不再被使用。
    pub unsafe fn reclaim(self) -> Vec<f32> {
        if self.ptr.is_null() {
            return Vec::new();
        }
        unsafe { Vec::from_raw_parts(self.ptr, self.len as usize, self.len as usize) }
    }
}

/// 把变长形状补进定长 `[u32; MAX_RANK]`。
pub fn pad_dims(dims: &[u32]) -> [u32; MAX_RANK] {
    let mut out = [0u32; MAX_RANK];
    out[..dims.len().min(MAX_RANK)].copy_from_slice(&dims[..dims.len().min(MAX_RANK)]);
    out
}

/// 逐元素核编号（宿主按 `mightbe-core::api::ElemOp` 映射到同一批常量）。
pub mod elem {
    pub const NEG: u32 = 0;
    pub const SIGMOID: u32 = 1;
    pub const TANH: u32 = 2;
    pub const RELU: u32 = 3;
    pub const GELU: u32 = 4;
    pub const EXP: u32 = 5;
    pub const LOG: u32 = 6;
    pub const SQRT: u32 = 7;
    pub const SQUARE: u32 = 8;
}

/// 宿主注入的能力表。插件优先复用这些核，而不是各自重写算子。
///
/// 字段一律 `Option`：新版本追加核时，老插件按偏移读到 `None` 只会拒绝调用，
/// 不会跳到错位的函数上。
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct HostApi {
    pub version: u32,
    pub alloc: Option<unsafe extern "C" fn(len: u64) -> *mut f32>,
    pub free: Option<unsafe extern "C" fn(ptr: *mut f32, len: u64)>,
    /// 行主序 `(m,k) x (k,n) -> (m,n)`，写入调用方提供的 `out`（长度 m*n）。
    pub matmul: Option<
        unsafe extern "C" fn(a: *const f32, b: *const f32, out: *mut f32, m: u32, k: u32, n: u32) -> i32,
    >,
    /// 就地逐元素核，返回 0 表示成功。
    pub elementwise: Option<unsafe extern "C" fn(out: *mut f32, len: u64, op: u32) -> i32>,
}

// ───────────────────────── 插件侧安全视图 ─────────────────────────

/// 插件方法收到的输入视图（零拷贝借用宿主内存）。
#[derive(Debug, Clone, Copy)]
pub struct Arg<'a> {
    pub data: &'a [f32],
    pub dims: &'a [u32],
}

/// # Safety
/// `t` 指向的 `HostTensor` 及其数据缓冲区必须在借用有效期内保持有效。
unsafe fn arg_view<'a>(t: &'a HostTensor) -> Arg<'a> {
    let data = if t.ptr.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(t.ptr, t.len as usize) }
    };
    Arg { data, dims: &t.dims[..t.ndim.min(MAX_RANK as u32) as usize] }
}

/// # Safety
/// `args[..n]` 必须是宿主构造的有效 `HostTensor` 数组。
unsafe fn arg_vec<'a>(args: *const HostTensor, n: u32) -> Vec<Arg<'a>> {
    if args.is_null() {
        return Vec::new();
    }
    let raw: &'a [HostTensor] = unsafe { std::slice::from_raw_parts(args, n as usize) };
    raw.iter().map(|t| unsafe { arg_view(t) }).collect()
}

impl HostTensor {
    /// 宿主侧构造只读视图。`dims` 长度须 ≤ [`MAX_RANK`]。
    pub fn view(data: &[f32], dims: &[u32]) -> HostTensor {
        debug_assert!(dims.len() <= MAX_RANK, "HostTensor::view 秩超上限");
        HostTensor {
            ptr: data.as_ptr(),
            len: data.len() as u64,
            ndim: dims.len() as u32,
            dims: pad_dims(dims),
        }
    }
}

/// 插件返回的张量：所有权在插件，SDK 复制进宿主内存。
#[derive(Debug, Clone)]
pub struct Out {
    pub data: Vec<f32>,
    pub dims: Vec<u32>,
}

impl Out {
    pub fn new(data: Vec<f32>, dims: Vec<u32>) -> Result<Self, String> {
        if dims.len() > MAX_RANK {
            return Err(format!("秩 {} 超过 ABI 上限 {MAX_RANK}", dims.len()));
        }
        let n: usize = dims.iter().map(|&d| d as usize).product();
        if n != data.len() {
            return Err(format!("形状 {dims:?} 与长度 {} 不符", data.len()));
        }
        Ok(Self { data, dims })
    }

    /// 一维产出。
    pub fn vec1(data: Vec<f32>) -> Self {
        let n = data.len();
        Self { data, dims: vec![n as u32] }
    }
}

/// 宿主能力的安全包装（传给插件的 `host` 参数）。
pub struct Host<'a> {
    api: &'a HostApi,
}

impl<'a> Host<'a> {
    /// # Safety
    /// `api` 必须来自宿主，且其函数指针有效。
    pub unsafe fn from_api(api: &'a HostApi) -> Self {
        Self { api }
    }

    /// 行主序矩阵乘。
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String> {
        if a.len() != m * k || b.len() != k * n {
            return Err(format!("matmul: 数据长度与 ({m},{k})x({k},{n}) 不符"));
        }
        let f = self.api.matmul.ok_or("宿主未提供 matmul")?;
        let mut out = vec![0f32; m * n];
        let rc = unsafe { f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m as u32, k as u32, n as u32) };
        if rc != 0 {
            return Err(format!("宿主 matmul 失败: rc={rc}"));
        }
        Ok(out)
    }

    /// 就地逐元素核。
    pub fn elementwise(&self, data: &mut [f32], op: u32) -> Result<(), String> {
        let f = self.api.elementwise.ok_or("宿主未提供 elementwise")?;
        let rc = unsafe { f(data.as_mut_ptr(), data.len() as u64, op) };
        if rc == 0 {
            Ok(())
        } else {
            Err(format!("宿主 elementwise 失败: rc={rc}"))
        }
    }

    fn alloc(&self, len: usize) -> Result<*mut f32, String> {
        let f = self.api.alloc.ok_or("宿主未提供 alloc")?;
        let ptr = unsafe { f(len as u64) };
        if ptr.is_null() {
            return Err("宿主 alloc 返回空指针".into());
        }
        Ok(ptr)
    }

    /// 把插件产出复制进宿主内存。
    fn publish(&self, out: &Out) -> Result<HostBuf, String> {
        let ptr = self.alloc(out.data.len())?;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, out.data.len()) };
        slice.copy_from_slice(&out.data);
        Ok(HostBuf {
            ptr,
            len: out.data.len() as u64,
            ndim: out.dims.len() as u32,
            dims: pad_dims(&out.dims),
        })
    }
}

fn write_slot(slot: *mut HostBuf, out: Option<Out>, host: &Host) -> Result<(), String> {
    let dst = unsafe { &mut *slot };
    *dst = match out {
        None => HostBuf::NIL,
        Some(v) => host.publish(&v)?,
    };
    Ok(())
}

// ───────────────────────── 插件 trait ─────────────────────────

/// 一个自定义层。除 `new` 外都要求可并发只读调用（`&self`），
/// 因此宿主可以让多个快照共享同一实例。
pub trait MbLayer: Send + Sync {
    /// 由 manifest/配置的 JSON 构造。参数形状在此定下，`params` 报告给宿主。
    fn new(config_json: &str) -> Result<Self, String>
    where
        Self: Sized;

    /// 权重声明：名称 + 形状。宿主据此分配、初始化并累积梯度；插件自己不留权重。
    fn params(&self) -> Vec<(String, Vec<u32>)>;

    /// 装载期静态形状推导（不跑前向）。v1 单输出。
    fn infer_shapes(&self, inputs: &[Vec<u32>]) -> Result<Vec<u32>, String>;

    /// 前向。`args` 顺序为「数据输入…，其后按 `params()` 顺序的参数」；
    /// 反向返回的梯度必须按同一顺序对齐。
    fn forward(&self, args: &[Arg], host: &Host) -> Result<Out, String>;

    /// 反向：给 `grad_out`（与 `out` 同形状）求各输入的梯度，`None` 表示该槽位无梯度。
    fn backward(
        &self,
        grad_out: &Arg,
        args: &[Arg],
        out: &Arg,
        host: &Host,
    ) -> Result<Vec<Option<Out>>, String>;
}

// ───────────────────────── 句柄与错误传递 ─────────────────────────

/// C ABI 句柄指向的堆上装箱：`*mut c_void` 是 thin 指针，装箱内才是 `dyn` 的 fat 指针。
pub struct Handle(pub Box<dyn MbLayer>);

impl Handle {
    pub fn into_raw(self) -> *mut c_void {
        Box::into_raw(Box::new(self)) as *mut c_void
    }

    /// # Safety
    /// `h` 必须是 [`Handle::into_raw`] 产出、且尚未 [`Handle::from_raw`] 的指针；
    /// 返回借用需在有效期内不再 drop。
    pub unsafe fn layer<'a>(h: *mut c_void) -> &'a dyn MbLayer {
        let boxed: &'a Box<dyn MbLayer> = unsafe { &(*h.cast::<Handle>()).0 };
        &**boxed
    }

    /// # Safety
    /// 同 [`Handle::layer`]，且此后该指针失效。
    pub unsafe fn from_raw(h: *mut c_void) -> Box<Handle> {
        unsafe { Box::from_raw(h as *mut Handle) }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// 记录本次调用的失败原因，宿主随后经 `mb_last_error` 取回。
pub fn set_error(msg: impl AsRef<str>) {
    let c = CString::new(msg.as_ref().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// [`mb_last_error`] 的实现：返回的指针在本线程下一次插件调用前有效。
pub fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// 读回本线程最近一次插件错误（宿主侧调试用）。
pub fn last_error_string() -> String {
    let p = last_error_ptr();
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}


/// 由泛型层类型构造擦除实例，`mb_export!` 把它排进构造函数表。
pub fn construct<T: MbLayer + 'static>(config_json: &str) -> Result<Handle, String> {
    Ok(Handle(Box::new(T::new(config_json)?)))
}

// ───────────────────────── 导出宏 ─────────────────────────

/// 生成 `mb_*` 导出符号。每个 cdylib 调用一次：
///
/// ```ignore
/// mb_export! {
///     api = mightbe_plugin::MB_ABI_VERSION,
///     layers = [
///         "dense_tanh" => DenseTanh,
///         "boom" => Boom,
///     ],
/// }
/// ```
#[macro_export]
macro_rules! mb_export {
    (
        api = $api:expr,
        layers = [ $( $name:literal => $ty:ty ),* $(,)? ],
    ) => {
        const MB_NAMES: &[&str] = &[ $( $name ),* ];
        /// NUL 结尾的层名表：`mb_describe` 直接回它的指针（'static，无需释放）。
        ///
        /// 注意不要写成 `concat!(...).as_bytes()`：带方法调用的表达式不做常量提升，
        /// 会在 const 里报 E0716，因此这里先攒 `&str` 字面量、用到时再取字节。
        const MB_NAMES_C: &[&str] = &[ $( concat!($name, "\0") ),* ];
        const MB_CTORS: &[fn(&str) -> ::std::result::Result<
            $crate::Handle,
            ::std::string::String,
        >] = &[ $( $crate::construct::<$ty> ),* ];

        #[no_mangle]
        pub extern "C" fn mb_api_version() -> u32 {
            $api
        }

        #[no_mangle]
        pub extern "C" fn mb_layer_count() -> u32 {
            MB_NAMES.len() as u32
        }

        /// 第 `i` 个层的类型名（NUL 结尾静态字符串）；越界返回空指针。
        #[no_mangle]
        pub extern "C" fn mb_describe(i: u32) -> *const ::std::ffi::c_char {
            match MB_NAMES_C.get(i as usize) {
                Some(s) => s.as_ptr() as *const ::std::ffi::c_char,
                None => ::std::ptr::null(),
            }
        }

        /// 按层序号 + 配置 JSON 建立实例；失败返回空句柄。
        #[no_mangle]
        pub unsafe extern "C" fn mb_create(
            i: u32,
            config_json: *const ::std::ffi::c_char,
        ) -> *mut ::std::ffi::c_void {
            let r = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| -> $crate::PlumbResult {
                let Some(ctor) = MB_CTORS.get(i as usize) else {
                    $crate::set_error(format!("mb_create: 层序号 {i} 不存在"));
                    return Err(());
                };
                let name = MB_NAMES[i as usize];
                let cfg: &str = if config_json.is_null() {
                    "{}"
                } else {
                    match unsafe { ::std::ffi::CStr::from_ptr(config_json) }.to_str() {
                        Ok(s) => s,
                        Err(_) => {
                            $crate::set_error(format!("层 {name}: config 不是合法 UTF-8"));
                            return Err(());
                        }
                    }
                };
                match ctor(cfg) {
                    Ok(h) => Ok(h.into_raw()),
                    Err(e) => {
                        $crate::set_error(format!("层 {name}: {e}"));
                        Err(())
                    }
                }
            }));
            match r {
                Ok(Ok(ptr)) => ptr,
                Ok(Err(())) | Err(_) => {
                    if r.is_err() {
                        $crate::set_error("mb_create: 插件构造函数 panic");
                    }
                    ::std::ptr::null_mut()
                }
            }
        }

        /// 写第 `i` 个参数：名字进 `name_buf`（容量 `name_cap`），形状进 `dims`
        /// （容量 `$crate::MAX_RANK`）。返回秩；`-1` 表示槽位读完（正常结束），
        /// `-2` 表示该参数超出 ABI 容量（真错误）。
        #[no_mangle]
        pub unsafe extern "C" fn mb_param(
            handle: *mut ::std::ffi::c_void,
            i: u32,
            name_buf: *mut ::std::ffi::c_char,
            name_cap: u32,
            dims: *mut u32,
        ) -> i32 {
            let r = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| -> i32 {
                let layer = unsafe { $crate::Handle::layer(handle) };
                // 必须先绑住 `params()` 的结果：`Vec` 是临时值，若在 `let` 里直接
                // `.get()`，它会在语句末尾 drop，后面再用 `name`/`shape` 就报 E0716。
                let params = layer.params();
                let Some((name, shape)) = params.get(i as usize) else {
                    return -1;
                };
                if shape.len() > $crate::MAX_RANK || name.len() + 1 > name_cap as usize {
                    $crate::set_error(format!("参数 {name}: 名字或形状超出 ABI 容量"));
                    return -2;
                }
                unsafe {
                    ::std::ptr::copy_nonoverlapping(
                        name.as_ptr(),
                        name_buf as *mut u8,
                        name.len(),
                    );
                    *name_buf.add(name.len()) = 0;
                    ::std::ptr::copy_nonoverlapping(shape.as_ptr(), dims, shape.len());
                }
                shape.len() as i32
            }));
            match r {
                Ok(v) => v,
                Err(_) => {
                    $crate::set_error("mb_param: 插件 panic");
                    -1
                }
            }
        }

        /// 静态形状推导。`ndims[k]` 是第 k 个输入的秩，`flat` 按序拼接各输入形状，
        /// 输出写成 `out_dims[0] = 秩`、`out_dims[1..] = 各轴`。
        /// 返回输出秩；失败返回 -1。
        #[no_mangle]
        pub unsafe extern "C" fn mb_infer_shapes(
            handle: *mut ::std::ffi::c_void,
            ndims: *const u32,
            flat: *const u32,
            n_inputs: u32,
            out_dims: *mut u32,
        ) -> i32 {
            let r = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| -> i32 {
                let layer = unsafe { $crate::Handle::layer(handle) };
                let mut ins = ::std::vec::Vec::with_capacity(n_inputs as usize);
                let mut off = 0usize;
                for k in 0..n_inputs as usize {
                    let d = unsafe { *ndims.add(k) } as usize;
                    if d > $crate::MAX_RANK {
                        $crate::set_error(format!("第 {k} 个输入秩 {d} 超上限"));
                        return -1;
                    }
                    ins.push(unsafe {
                        ::std::slice::from_raw_parts(flat.add(off), d)
                    }.to_vec());
                    off += d;
                }
                match layer.infer_shapes(&ins) {
                    Ok(shape) if shape.len() <= $crate::MAX_RANK => {
                        let n = shape.len();
                        unsafe {
                            *out_dims = n as u32;
                            ::std::ptr::copy_nonoverlapping(shape.as_ptr(), out_dims.add(1), n);
                        }
                        n as i32
                    }
                    Ok(shape) => {
                        $crate::set_error(format!("输出秩 {} 超上限", shape.len()));
                        -1
                    }
                    Err(e) => {
                        $crate::set_error(e);
                        -1
                    }
                }
            }));
            match r {
                Ok(v) => v,
                Err(_) => {
                    $crate::set_error("mb_infer_shapes: 插件 panic");
                    -1
                }
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn mb_forward(
            handle: *mut ::std::ffi::c_void,
            args: *const $crate::HostTensor,
            n_args: u32,
            host: *const $crate::HostApi,
            out: *mut $crate::HostBuf,
        ) -> i32 {
            if host.is_null() || out.is_null() {
                $crate::set_error("mb_forward: host/out 为空");
                return -3;
            }
            let r = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| -> i32 {
                let api = unsafe { &*host };
                let layer = unsafe { $crate::Handle::layer(handle) };
                let view = $crate::__arg_vec(args, n_args);
                let h = unsafe { $crate::Host::from_api(api) };
                match layer.forward(&view, &h) {
                    Ok(v) => match h.publish_out(&v) {
                        Ok(buf) => {
                            unsafe { *out = buf };
                            0
                        }
                        Err(e) => {
                            $crate::set_error(e);
                            -1
                        }
                    },
                    Err(e) => {
                        $crate::set_error(e);
                        -1
                    }
                }
            }));
            match r {
                Ok(v) => v,
                Err(_) => {
                    $crate::set_error("mb_forward: 插件 panic（已被 SDK 隔离）");
                    -2
                }
            }
        }

        /// `dargs` 长度须等于 `n_args`；无梯度的槽位写空缓冲。
        #[no_mangle]
        pub unsafe extern "C" fn mb_backward(
            handle: *mut ::std::ffi::c_void,
            grad_out: *const $crate::HostTensor,
            args: *const $crate::HostTensor,
            n_args: u32,
            out: *const $crate::HostTensor,
            dargs: *mut $crate::HostBuf,
            host: *const $crate::HostApi,
        ) -> i32 {
            if host.is_null() || dargs.is_null() || grad_out.is_null() || out.is_null() {
                $crate::set_error("mb_backward: 参数指针为空");
                return -3;
            }
            let r = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| -> i32 {
                let api = unsafe { &*host };
                let layer = unsafe { $crate::Handle::layer(handle) };
                let view = $crate::__arg_vec(args, n_args);
                let go = $crate::__arg_view(grad_out);
                let oy = $crate::__arg_view(out);
                let h = unsafe { $crate::Host::from_api(api) };
                match layer.backward(&go, &view, &oy, &h) {
                    Err(e) => {
                        $crate::set_error(e);
                        -1
                    }
                    Ok(grads) if grads.len() != n_args as usize => {
                        $crate::set_error(format!(
                            "backward 返回 {} 个梯度，期望 {n_args} 个",
                            grads.len()
                        ));
                        -1
                    }
                    Ok(grads) => {
                        for (i, g) in grads.into_iter().enumerate() {
                            if let Err(e) = $crate::__write_slot(dargs.add(i), g, &h) {
                                $crate::set_error(e);
                                return -1;
                            }
                        }
                        0
                    }
                }
            }));
            match r {
                Ok(v) => v,
                Err(_) => {
                    $crate::set_error("mb_backward: 插件 panic（已被 SDK 隔离）");
                    -2
                }
            }
        }

        #[no_mangle]
        pub unsafe extern "C" fn mb_drop(handle: *mut ::std::ffi::c_void) {
            if handle.is_null() {
                return;
            }
            let _ = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                drop(unsafe { $crate::Handle::from_raw(handle) });
            }));
        }

        /// 本线程最近一次失败的说明；指针在下次插件调用前有效。
        #[no_mangle]
        pub extern "C" fn mb_last_error() -> *const ::std::ffi::c_char {
            $crate::last_error_ptr()
        }
    };
}

/// `mb_export!` 内部用的桥接类型。
#[doc(hidden)]
pub type PlumbResult = Result<*mut c_void, ()>;

/// `mb_export!` 内部用。
///
/// # Safety
/// 见 [`arg_vec`]。
#[doc(hidden)]
pub unsafe fn __arg_vec<'a>(args: *const HostTensor, n: u32) -> Vec<Arg<'a>> {
    unsafe { arg_vec(args, n) }
}

/// `mb_export!` 内部用。
///
/// # Safety
/// 见 [`arg_view`]。
#[doc(hidden)]
pub unsafe fn __arg_view<'a>(t: *const HostTensor) -> Arg<'a> {
    unsafe { arg_view(&*t) }
}

/// `mb_export!` 内部用。
///
/// # Safety
/// `slot` 必须有效。
#[doc(hidden)]
pub unsafe fn __write_slot(
    slot: *mut HostBuf,
    out: Option<Out>,
    host: &Host,
) -> Result<(), String> {
    write_slot(slot, out, host)
}

impl<'a> Host<'a> {
    /// `mb_export!` 内部用：把产出复制进宿主内存。
    #[doc(hidden)]
    pub fn publish_out(&self, out: &Out) -> Result<HostBuf, String> {
        self.publish(out)
    }
}

#[cfg(test)]
mod macro_smoke {
    use super::*;

    struct Probe;

    impl MbLayer for Probe {
        fn new(_cfg: &str) -> Result<Self, String> {
            Ok(Probe)
        }
        fn params(&self) -> Vec<(String, Vec<u32>)> {
            vec![("w".into(), vec![2u32, 3u32])]
        }
        fn infer_shapes(&self, _i: &[Vec<u32>]) -> Result<Vec<u32>, String> {
            Ok(vec![2, 3])
        }
        fn forward(&self, a: &[Arg], _h: &Host) -> Result<Out, String> {
            Ok(Out::vec1(a[0].data.to_vec()))
        }
        fn backward(&self, _g: &Arg, _a: &[Arg], _o: &Arg, _h: &Host) -> Result<Vec<Option<Out>>, String> {
            Ok(vec![None])
        }
    }

    mb_export! {
        api = MB_ABI_VERSION,
        layers = ["probe" => Probe],
    }
}
