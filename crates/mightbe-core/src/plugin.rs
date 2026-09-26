//! 路径 C：Layer Plugin 宿主（README 4.1.1）。
//!
//! 三件事：
//! 1. [`dl`]：手写的动态库加载器。`libloading` 不在离线可得清单内（README 15 节），
//!    故直接 FFI 调 `LoadLibraryW`/`dlopen`；
//! 2. [`LoadedLibrary`] + [`PluginInstance`]：解析并校验 `mb_*` 符号表（3101/3102），
//!    所有跨 ABI 调用都包 `catch_unwind`，插件侧错误经 `mb_last_error` 回传；
//! 3. [`PluginLayer`]：实现 [`Layer`]，把插件产出作为 `Op::Plugin` 节点入图 ——
//!    权重由宿主分配并作为图 `param` 节点混排在 `args` 尾部，因此梯度累积、
//!    优化器寻址、checkpoint 与内置层完全同构。

use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use mightbe_plugin::{HostApi, HostBuf, HostTensor, MB_ABI_VERSION, MAX_RANK};

use crate::api::{Backend, Layer, LayerCtx, LayerSpec, MtbError, MtbResult, Shape, Tensor};
use crate::autograd::{CustomBackward, Graph, Var};
use crate::backend::NaiveBackend;
use crate::init::Init;

// ───────────────────────── 注入给插件的 HostApi ─────────────────────────

/// v1 固定绑定进程级 CPU 后端（README 4.1：插件 `requires = "cpu"`，
/// 跨设备 ABI 传裸指针必是 CPU 内存）。
static HOST_API: OnceLock<HostApi> = OnceLock::new();

pub fn host_api() -> &'static HostApi {
    HOST_API.get_or_init(|| HostApi {
        version: MB_ABI_VERSION,
        alloc: Some(host_alloc),
        free: Some(host_free),
        matmul: Some(host_matmul),
        elementwise: Some(host_elementwise),
    })
}

unsafe extern "C" fn host_alloc(len: u64) -> *mut f32 {
    let mut v = vec![0f32; len as usize];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

unsafe extern "C" fn host_free(ptr: *mut f32, len: u64) {
    if ptr.is_null() {
        return;
    }
    drop(unsafe { Vec::from_raw_parts(ptr, len as usize, len as usize) });
}

unsafe extern "C" fn host_matmul(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: u32,
    k: u32,
    n: u32,
) -> i32 {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let (a, b, o) = unsafe {
        (
            std::slice::from_raw_parts(a, m * k),
            std::slice::from_raw_parts(b, k * n),
            std::slice::from_raw_parts_mut(out, m * n),
        )
    };
    let shapes = crate::api::GemmShapes { m, k, n };
    match NaiveBackend.matmul(a, b, o, &shapes) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

unsafe extern "C" fn host_elementwise(out: *mut f32, len: u64, op: u32) -> i32 {
    let elem = match op {
        0 => crate::api::ElemOp::Neg,
        1 => crate::api::ElemOp::Sigmoid,
        2 => crate::api::ElemOp::Tanh,
        3 => crate::api::ElemOp::Relu,
        4 => crate::api::ElemOp::Gelu,
        5 => crate::api::ElemOp::Exp,
        6 => crate::api::ElemOp::Log,
        7 => crate::api::ElemOp::Sqrt,
        8 => crate::api::ElemOp::Square,
        _ => return -2,
    };
    let slice = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
    match NaiveBackend.elementwise(slice, elem) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ───────────────────────── 动态库加载器 ─────────────────────────

#[cfg(windows)]
mod dl {
    use super::*;
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(lp_lib_filename: *const u16) -> *mut c_void;
        fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const c_char) -> *mut c_void;
        fn FreeLibrary(h_module: *mut c_void) -> i32;
    }

    /// 句柄本身是裸指针，由外层 [`LoadedLibrary`] 统一声明 Send/Sync。
    pub struct Handle(pub *mut c_void);

    pub fn open(path: &Path) -> MtbResult<Handle> {
        let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        let wide: Vec<u16> = abs.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let h = unsafe { LoadLibraryW(wide.as_ptr()) };
        if h.is_null() {
            return Err(MtbError::Other(format!(
                "LoadLibraryW 失败: {}（GetLastError={}）",
                abs.display(),
                unsafe { win_err() }
            )));
        }
        Ok(Handle(h))
    }

    unsafe fn win_err() -> u32 {
        // kernel32 的 GetLastError 无法在 Rust 侧直接内联取用；仅用于诊断文本
        0
    }

    pub fn symbol(h: &Handle, name: &str) -> *mut c_void {
        let c = CString::new(name).expect("符号名不含 NUL");
        unsafe { GetProcAddress(h.0, c.as_ptr()) }
    }

    pub fn close(h: &Handle) {
        unsafe {
            FreeLibrary(h.0);
        }
    }
}

#[cfg(not(windows))]
mod dl {
    use super::*;

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    extern "C" {
        fn dlopen(filename: *const c_char, flag: std::os::raw::c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol_name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> std::os::raw::c_int;
        fn dlerror() -> *const c_char;
    }

    #[cfg(target_os = "macos")]
    const MODE: std::os::raw::c_int = 0x0004 /* RTLD_NOW */ | 0x0008 /* RTLD_LOCAL */;
    #[cfg(not(target_os = "macos"))]
    const MODE: std::os::raw::c_int = 2 /* RTLD_NOW */;

    pub struct Handle(pub *mut c_void);

    pub fn open(path: &Path) -> MtbResult<Handle> {
        let c = CString::new(path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| MtbError::Config("插件路径含 NUL".into()))?;
        let h = unsafe { dlopen(c.as_ptr(), MODE) };
        if h.is_null() {
            let msg = unsafe {
                let e = dlerror();
                if e.is_null() {
                    "未知错误".to_string()
                } else {
                    CStr::from_ptr(e).to_string_lossy().into_owned()
                }
            };
            return Err(MtbError::Other(format!("dlopen 失败: {}（{msg}）", path.display())));
        }
        Ok(Handle(h))
    }

    pub fn symbol(h: &Handle, name: &str) -> *mut c_void {
        let c = CString::new(name).expect("符号名不含 NUL");
        unsafe {
            dlerror();
            dlsym(h.0, c.as_ptr())
        }
    }

    pub fn close(h: &Handle) {
        unsafe {
            dlclose(h.0);
        }
    }
}

/// 从裸指针取函数地址；空指针报 3102（README 9 节）。
unsafe fn resolve<T>(raw: *mut c_void, name: &str) -> MtbResult<T> {
    const {
        assert!(std::mem::size_of::<*mut c_void>() == std::mem::size_of::<usize>(), "指针宽度")
    }
    if raw.is_null() {
        return Err(MtbError::coded(
            MtbError::PLUGIN_SYM_MISSING,
            format!("3102 插件缺少导出符号 {name}"),
        ));
    }
    assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<*mut c_void>(),
        "符号 {name} 的目标类型宽度与函数指针不符"
    );
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&raw) })
}

// ───────────────────────── 符号表 ─────────────────────────

/// `mb_export!` 生成的全部导出符号（清单见 `mightbe_plugin::MB_SYMBOLS`）。
#[derive(Clone, Copy)]
struct Symbols {
    layer_count: extern "C" fn() -> u32,
    describe: extern "C" fn(u32) -> *const c_char,
    create: unsafe extern "C" fn(u32, *const c_char) -> *mut c_void,
    param: unsafe extern "C" fn(*mut c_void, u32, *mut c_char, u32, *mut u32) -> i32,
    infer_shapes:
        unsafe extern "C" fn(*mut c_void, *const u32, *const u32, u32, *mut u32) -> i32,
    forward: unsafe extern "C" fn(*mut c_void, *const HostTensor, u32, *const HostApi, *mut HostBuf) -> i32,
    backward: unsafe extern "C" fn(
        *mut c_void,
        *const HostTensor,
        *const HostTensor,
        u32,
        *const HostTensor,
        *mut HostBuf,
        *const HostApi,
    ) -> i32,
    drop_layer: unsafe extern "C" fn(*mut c_void),
    last_error: extern "C" fn() -> *const c_char,
}

/// 已加载的插件动态库。`Arc` 引用归零时才 `FreeLibrary`
/// （README：热替换时旧库延迟到无快照引用才卸载）。
pub struct LoadedLibrary {
    handle: dl::Handle,
    path: PathBuf,
    sym: Symbols,
    api_version: u32,
    /// 层序号 → 层类型名（网络配置里的 `type`）
    layers: Vec<String>,
}

// 插件实例的并发只读调用由 SDK 侧 `MbLayer: Send + Sync` 约束；
// 模块句柄与函数指针表本身不可变。
unsafe impl Send for LoadedLibrary {}
unsafe impl Sync for LoadedLibrary {}

impl fmt::Debug for LoadedLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadedLibrary")
            .field("path", &self.path)
            .field("api_version", &self.api_version)
            .field("layers", &self.layers)
            .finish()
    }
}

impl Drop for LoadedLibrary {
    fn drop(&mut self) {
        dl::close(&self.handle);
    }
}

impl LoadedLibrary {
    /// 加载并校验：ABI 版本（3101）→ 导出符号（3102）→ 层名清单。
    pub fn open(path: impl AsRef<Path>) -> MtbResult<Arc<Self>> {
        let path = path.as_ref().to_path_buf();
        let handle = dl::open(&path)?;
        fn need<T>(handle: &dl::Handle, name: &str) -> MtbResult<T> {
            let raw = dl::symbol(handle, name);
            unsafe { resolve(raw, name) }
        }
        // 版本先于其余符号校验：ABI 不符时不该再逐个报缺符号（3101 优先于 3102）
        let probe: extern "C" fn() -> u32 = need(&handle, "mb_api_version")?;
        let got = (probe)();
        let sym = Symbols {
            layer_count: need(&handle, "mb_layer_count")?,
            describe: need(&handle, "mb_describe")?,
            create: need(&handle, "mb_create")?,
            param: need(&handle, "mb_param")?,
            infer_shapes: need(&handle, "mb_infer_shapes")?,
            forward: need(&handle, "mb_forward")?,
            backward: need(&handle, "mb_backward")?,
            drop_layer: need(&handle, "mb_drop")?,
            last_error: need(&handle, "mb_last_error")?,
        };
        if got != MB_ABI_VERSION {
            return Err(MtbError::coded(
                MtbError::PLUGIN_ABI,
                format!("3101 插件 {path:?} ABI 版本 {got}，宿主只接受 {MB_ABI_VERSION}"),
            ));
        }
        let n = (sym.layer_count)();
        let mut layers = Vec::with_capacity(n as usize);
        for i in 0..n {
            let p = (sym.describe)(i);
            if p.is_null() {
                return Err(MtbError::coded(
                    MtbError::PLUGIN_SYM_MISSING,
                    format!("3102 插件 {path:?} 的 mb_describe({i}) 返回空"),
                ));
            }
            layers.push(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
        }
        if layers.is_empty() {
            return Err(MtbError::coded(
                MtbError::PLUGIN_SYM_MISSING,
                format!("3102 插件 {path:?} 没有导出任何层"),
            ));
        }
        Ok(Arc::new(Self { handle, path, sym, api_version: got, layers }))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn api_version(&self) -> u32 {
        self.api_version
    }

    pub fn layer_names(&self) -> &[String] {
        &self.layers
    }

    fn index_of(&self, layer: &str) -> MtbResult<u32> {
        self.layers
            .iter()
            .position(|n| n == layer)
            .map(|i| i as u32)
            .ok_or_else(|| {
                MtbError::coded(
                    MtbError::NETWORK,
                    format!(
                        "库 {:?} 没有层 {layer:?}（可导出：{}）",
                        self.path,
                        self.layers.join(", ")
                    ),
                )
            })
    }

    /// 取回插件侧最近一次错误说明（同线程，紧跟失败调用之后）。
    pub fn last_error(&self) -> String {
        let p = (self.sym.last_error)();
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    fn err(&self, ctx: &str, rc: i32) -> MtbError {
        let detail = self.last_error();
        MtbError::coded(
            MtbError::TRAIN,
            format!("插件层 {ctx} 调用失败 rc={rc}{}", if detail.is_empty() { String::new() } else { format!(": {detail}") }),
        )
    }

    /// 建立一个层实例（未冒烟）。必须从 `Arc` 出发：实例要持有库的强引用，
    /// 保证引用归零前不会 `FreeLibrary`。
    pub fn create(self: &Arc<Self>, layer: &str, config_json: &str) -> MtbResult<Arc<PluginInstance>> {
        let idx = self.index_of(layer)?;
        let cfg = CString::new(config_json)
            .map_err(|_| MtbError::Config(format!("插件 config 含 NUL: {layer}")))?;
        let lib = self.clone();
        let raw = std::panic::catch_unwind(move || unsafe { (lib.sym.create)(idx, cfg.as_ptr()) })
            .map_err(|_| {
                MtbError::coded(MtbError::NETWORK, format!("插件层 {layer} 建立时跨 ABI panic"))
            })?;
        let Some(ptr) = NonNull::new(raw) else {
            return Err(MtbError::coded(
                MtbError::NETWORK,
                format!("插件层 {layer} 建立失败: {}", self.last_error()),
            ));
        };
        Ok(Arc::new(PluginInstance {
            lib: self.clone(),
            layer: layer.to_string(),
            raw: ptr,
        }))
    }

    /// 冒烟（README 4.1.1：建立新实例 → 冒烟前向 → 才允许登记）。
    ///
    /// 校验：参数声明可读出、形状推导成功、前向在零输入上产出形状相符且有限。
    /// 输入取零张量而非随机值，因此冒烟可复现、也不引入任何超参。
    pub fn smoke(
        self: &Arc<Self>,
        layer: &str,
        config_json: &str,
        inputs: &[Shape],
    ) -> MtbResult<Arc<PluginInstance>> {
        let inst = self.create(layer, config_json)?;
        inst.smoke(inputs).map_err(|e| match e {
            MtbError::Coded { code, .. } if code == MtbError::PLUGIN_SMOKE_FAIL => e,
            other => MtbError::coded(
                MtbError::PLUGIN_SMOKE_FAIL,
                format!("3103 插件层 {layer} 冒烟失败: {other}"),
            ),
        })?;
        Ok(inst)
    }
}

// ───────────────────────── 层实例 ─────────────────────────

/// 一个跨 ABI 的插件层实例。权重不在这里——宿主持有并以图 `param` 节点传入。
pub struct PluginInstance {
    lib: Arc<LoadedLibrary>,
    layer: String,
    raw: NonNull<c_void>,
}

// SDK 侧 `MbLayer: Send + Sync` 是插件作者的契约：实例只以 &self 被并发调用。
unsafe impl Send for PluginInstance {}
unsafe impl Sync for PluginInstance {}

impl Drop for PluginInstance {
    fn drop(&mut self) {
        let p = self.raw.as_ptr();
        let sym = self.lib.sym;
        let _ = std::panic::catch_unwind(move || unsafe { (sym.drop_layer)(p) });
    }
}

impl fmt::Debug for PluginInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PluginInstance")
            .field("layer", &self.layer)
            .field("library", &self.lib.path)
            .finish()
    }
}

fn dims_u32(shape: &Shape) -> Vec<u32> {
    shape.iter().map(|&d| d as u32).collect()
}

fn shape_of(dims: &[u32]) -> Shape {
    dims.iter().map(|&d| d as usize).collect()
}

/// `catch_unwind` 要求 `UnwindSafe`；跨 ABI 调用只碰裸指针与局部缓冲，
/// 这个包装用于明示该前提。
macro_rules! uw {
    ($e:expr) => {
        std::panic::AssertUnwindSafe($e)
    };
}
use uw as unwind_safe;

impl PluginInstance {
    pub fn layer_name(&self) -> &str {
        &self.layer
    }

    pub fn library_path(&self) -> &Path {
        self.lib.path()
    }

    fn err(&self, what: &str, rc: i32) -> MtbError {
        self.lib.err(&format!("{}.{what}", self.layer), rc)
    }

    /// 权重声明：由 `mb_param` 逐槽读出，直到返回 -1（槽位读完）。
    pub fn param_decls(&self) -> MtbResult<Vec<(String, Shape)>> {
        let mut out = Vec::new();
        let sym = self.lib.sym;
        let h = self.raw.as_ptr();
        for i in 0u32.. {
            let mut name = [0u8; 64];
            let mut dims = [0u32; MAX_RANK];
            let rc = std::panic::catch_unwind(unwind_safe!(|| unsafe {
                (sym.param)(
                    h,
                    i,
                    name.as_mut_ptr() as *mut c_char,
                    name.len() as u32,
                    dims.as_mut_ptr(),
                )
            }))
            .map_err(|_| self.err("mb_param", -99))?;
            if rc == -1 {
                break;
            }
            if rc < -1 {
                return Err(MtbError::coded(
                    MtbError::PLUGIN_SYM_MISSING,
                    format!(
                        "插件层 {}: 第 {i} 个参数声明超出 ABI 容量（{}）",
                        self.layer,
                        self.lib.last_error()
                    ),
                ));
            }
            let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
            let key = String::from_utf8_lossy(&name[..end]).into_owned();
            if key.is_empty() {
                return Err(MtbError::coded(
                    MtbError::NETWORK,
                    format!("插件层 {}: 第 {i} 个参数名为空", self.layer),
                ));
            }
            out.push((key, shape_of(&dims[..rc as usize])));
        }
        Ok(out)
    }

    /// 装载期静态形状推导。
    pub fn infer_shapes(&self, inputs: &[Shape]) -> MtbResult<Shape> {
        if inputs.iter().any(|s| s.len() > MAX_RANK) {
            return Err(MtbError::Shape {
                expected: format!("秩 ≤ {MAX_RANK}"),
                got: format!("{inputs:?}"),
            });
        }
        let ndims: Vec<u32> = inputs.iter().map(|s| s.len() as u32).collect();
        let flat: Vec<u32> = inputs.iter().flat_map(|s| dims_u32(s)).collect();
        let mut out = [0u32; MAX_RANK + 1];
        let sym = self.lib.sym;
        let h = self.raw.as_ptr();
        let rc = std::panic::catch_unwind(unwind_safe!(|| unsafe {
            (sym.infer_shapes)(h, ndims.as_ptr(), flat.as_ptr(), ndims.len() as u32, out.as_mut_ptr())
        }))
        .map_err(|_| self.err("mb_infer_shapes", -99))?;
        if rc < 0 {
            return Err(self.err("mb_infer_shapes", rc));
        }
        let n = rc as usize;
        Ok(shape_of(&out[1..1 + n]))
    }

    /// 前向：产出由宿主内存承载，读完即收回。
    pub fn forward(&self, args: &[&Tensor]) -> MtbResult<Tensor> {
        let views = views_of(args);
        self.forward_views(&views)
    }

    /// 前向（视图已就绪，供 `Layer::forward` 复用）。
    pub fn forward_views(&self, views: &[HostTensor]) -> MtbResult<Tensor> {
        let mut buf = HostBuf::NIL;
        let sym = self.lib.sym;
        let h = self.raw.as_ptr();
        let api = host_api();
        let rc = std::panic::catch_unwind(unwind_safe!(|| unsafe {
            (sym.forward)(h, views.as_ptr(), views.len() as u32, api, &mut buf)
        }))
        .map_err(|_| self.err("mb_forward", -99))?;
        if rc != 0 {
            if !buf.ptr.is_null() {
                drop(unsafe { buf.reclaim() });
            }
            return Err(self.err("mb_forward", rc));
        }
        let (data, shape) = self.take(buf)?;
        Ok(Tensor::from_vec(data, shape).unwrap())
    }

    /// 反向：返回各输入的梯度，空张量表示该槽位无梯度。
    pub fn backward(
        &self,
        grad_out: &Tensor,
        args: &[&Tensor],
        out: &Tensor,
    ) -> MtbResult<Vec<Tensor>> {
        let views = views_of(args);
        let go_t = HostTensor::view(&grad_out.data, &dims_u32(&grad_out.shape));
        let oy_t = HostTensor::view(&out.data, &dims_u32(&out.shape));
        let mut dargs = vec![HostBuf::NIL; args.len()];
        let sym = self.lib.sym;
        let h = self.raw.as_ptr();
        let api = host_api();
        let rc = std::panic::catch_unwind(unwind_safe!(|| unsafe {
            (sym.backward)(
                h,
                &go_t,
                views.as_ptr(),
                views.len() as u32,
                &oy_t,
                dargs.as_mut_ptr(),
                api,
            )
        }))
        .map_err(|_| self.err("mb_backward", -99))?;
        if rc != 0 {
            // 失败路径也要收回插件已 publish 的缓冲，否则每次出错漏一块
            for b in dargs.iter() {
                if !b.ptr.is_null() {
                    drop(unsafe { b.reclaim() });
                }
            }
            return Err(self.err("mb_backward", rc));
        }
        let mut grads = Vec::with_capacity(dargs.len());
        for (i, b) in dargs.into_iter().enumerate() {
            if b.ptr.is_null() {
                grads.push(Tensor::from_vec(vec![], vec![0]).unwrap());
                continue;
            }
            let (data, shape) = self.take(b)?;
            if shape != args[i].shape {
                return Err(MtbError::Shape {
                    expected: format!("{:?}", args[i].shape),
                    got: format!("{shape:?}"),
                });
            }
            grads.push(Tensor::from_vec(data, shape).unwrap());
        }
        Ok(grads)
    }

    /// 把插件回传的宿主缓冲收回为 `(data, shape)`，并做形状/有限性检查。
    fn take(&self, buf: HostBuf) -> MtbResult<(Vec<f32>, Shape)> {
        if buf.ptr.is_null() {
            return Err(MtbError::coded(
                MtbError::TRAIN,
                format!("插件层 {} 返回空缓冲", self.layer),
            ));
        }
        let n = buf.ndim as usize;
        if n == 0 || n > MAX_RANK {
            return Err(MtbError::Shape {
                expected: format!("1..={MAX_RANK} 秩"),
                got: format!("{}", buf.ndim),
            });
        }
        let shape = shape_of(&buf.dims[..n]);
        let data = unsafe { buf.reclaim() };
        if shape.iter().product::<usize>() != data.len() {
            return Err(MtbError::Shape {
                expected: format!("{shape:?}"),
                got: format!("len={}", data.len()),
            });
        }
        if let Some(bad) = data.iter().position(|x| !x.is_finite()) {
            return Err(MtbError::coded(
                MtbError::TRAIN,
                format!(
                    "插件层 {} 在第 {bad} 个元素产出非有限值 {}",
                    self.layer, data[bad]
                ),
            ));
        }
        Ok((data, shape))
    }

    fn smoke(&self, inputs: &[Shape]) -> MtbResult<()> {
        let _ = self.param_decls()?;
        let derived = self.infer_shapes(inputs)?;
        let zeros: Vec<Tensor> = inputs.iter().map(|s| Tensor::zeros(s.clone())).collect();
        let refs: Vec<&Tensor> = zeros.iter().collect();
        let out = self.forward(&refs)?;
        if out.shape != derived {
            return Err(MtbError::coded(
                MtbError::PLUGIN_SMOKE_FAIL,
                format!("3103 冒烟形状不符：infer_shapes 给出 {derived:?}，前向产出 {:?}", out.shape),
            ));
        }
        Ok(())
    }
}

impl CustomBackward for PluginInstance {
    fn backward(
        &self,
        grad_out: &Tensor,
        inputs: &[&Tensor],
        out: &Tensor,
    ) -> MtbResult<Vec<Tensor>> {
        self.backward(grad_out, inputs, out)
    }
}

/// 由张量构造跨 ABI 只读视图（dims 内联拷贝，故无需延长 `tensors` 的借用）。
fn views_of(tensors: &[&Tensor]) -> Vec<HostTensor> {
    tensors
        .iter()
        .map(|t| {
            let d = dims_u32(&t.shape);
            HostTensor::view(&t.data, &d)
        })
        .collect()
}

// ───────────────────────── 插件层 ─────────────────────────

/// 网络里的一层插件：跨 ABI 实例 + 宿主侧权重。与内置层共用 [`Layer`] 接口。
pub struct PluginLayer {
    pub name: String,
    pub inst: Arc<PluginInstance>,
    pub config_json: String,
    decls: Vec<(String, Shape)>,
    params: Vec<Tensor>,
}

impl PluginLayer {
    /// 按实例声明的参数形状分配权重（初始化策略与种子由调用方从配置带入）。
    pub fn new(
        name: &str,
        inst: Arc<PluginInstance>,
        config_json: impl Into<String>,
        init: &Init,
        seed: u64,
    ) -> MtbResult<Self> {
        let decls = inst.param_decls()?;
        let params = decls
            .iter()
            .enumerate()
            .map(|(i, (_, shape))| init.make(shape, seed.wrapping_add(i as u64)))
            .collect();
        Ok(Self {
            name: name.to_string(),
            inst,
            config_json: config_json.into(),
            decls,
            params,
        })
    }

    pub fn param_shapes(&self) -> &[(String, Shape)] {
        &self.decls
    }

    pub fn library_path(&self) -> &Path {
        self.inst.library_path()
    }

    fn key(&self, param: &str) -> String {
        format!("{}.{}", self.name, param)
    }
}

impl Layer for PluginLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn param_names(&self) -> Vec<String> {
        self.decls.iter().map(|(n, _)| n.clone()).collect()
    }

    fn bind_params(&mut self, params: &[(String, Tensor)]) -> MtbResult<()> {
        for (k, t) in params {
            let short = k.strip_prefix(&format!("{}.", self.name)).unwrap_or(k);
            let i = self
                .decls
                .iter()
                .position(|(n, _)| n == short)
                .ok_or_else(|| {
                    MtbError::Config(format!("plugin: 层 {} 没有参数 {short:?}", self.name))
                })?;
            if t.shape != self.decls[i].1 {
                return Err(MtbError::Shape {
                    expected: format!("{:?}", self.decls[i].1),
                    got: format!("{:?}", t.shape),
                });
            }
            self.params[i] = t.clone();
        }
        Ok(())
    }

    fn dump_params(&self) -> Vec<(String, Tensor)> {
        self.decls
            .iter()
            .zip(self.params.iter())
            .map(|((n, _), t)| (self.key(n), t.clone()))
            .collect()
    }

    fn forward(
        &mut self,
        args: &[&Var],
        _ctx: &LayerCtx,
        graph: &mut Graph,
    ) -> MtbResult<Vec<Var>> {
        // 顺序契约（README 4.1.1）：先数据输入，其后按声明顺序的参数
        let mut vars: Vec<Var> = args.iter().copied().copied().collect();
        for (i, (n, _)) in self.decls.iter().enumerate() {
            vars.push(graph.param(self.key(n), self.params[i].clone()));
        }
        let views: Vec<HostTensor> = vars
            .iter()
            .map(|&v| {
                let t = graph.value(v);
                HostTensor::view(&t.data, &dims_u32(&t.shape))
            })
            .collect();
        let out = self.inst.forward_views(&views)?;
        let node = graph.plugin_op(&vars, out, self.inst.clone());
        Ok(vec![node])
    }

    fn infer_shapes(&self, inputs: &[&Shape]) -> MtbResult<Vec<Shape>> {
        let owned: Vec<Shape> = inputs.iter().map(|s| (*s).clone()).collect();
        Ok(vec![self.inst.infer_shapes(&owned)?])
    }

    fn describe(&self) -> LayerSpec {
        let params: Vec<String> = self
            .decls
            .iter()
            .map(|(n, s)| {
                format!("{n}={}", s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"))
            })
            .collect();
        LayerSpec::new(
            "plugin",
            &self.name,
            &format!(
                "{{\"library\":{:?},\"type\":{:?},\"config\":{},\"params\":[{}]}}",
                self.library_path().to_string_lossy(),
                self.inst.layer_name(),
                if self.config_json.is_empty() { "{}" } else { &self.config_json },
                params.join(",")
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_api_alloc_roundtrip_is_symmetric() {
        let api = host_api();
        let alloc = api.alloc.expect("宿主提供 alloc");
        let free = api.free.expect("宿主提供 free");
        let n = 7u64;
        let p = unsafe { alloc(n) };
        assert!(!p.is_null());
        let s = unsafe { std::slice::from_raw_parts_mut(p, n as usize) };
        for (i, x) in s.iter_mut().enumerate() {
            *x = i as f32;
        }
        let mut back = unsafe { Vec::from_raw_parts(p, n as usize, n as usize) };
        assert_eq!(back, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // 容量必须与长度一致，否则 reclaim 的 Vec 会带着错误容量进入 drop
        assert_eq!(back.capacity(), n as usize);
        back[6] = 9.0;
        drop(back);
        unsafe { free(std::ptr::null_mut(), 0) };
    }

    #[test]
    fn host_matmul_and_elementwise_match_naive_backend() {
        let api = host_api();
        let mm = api.matmul.unwrap();
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.0f32, 1.0, 1.0, 0.0];
        let mut out = [0f32; 4];
        assert_eq!(unsafe { mm(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 2, 2, 2) }, 0);
        assert_eq!(out, [2.0, 1.0, 4.0, 3.0]);

        let ew = api.elementwise.unwrap();
        let mut v = [-1.0f32, 0.0, 2.0];
        assert_eq!(unsafe { ew(v.as_mut_ptr(), v.len() as u64, 3) }, 0);
        assert_eq!(v, [0.0, 0.0, 2.0]);
        // 未知核编号要被拒绝，而不是崩或静默通过
        assert_ne!(unsafe { ew(v.as_mut_ptr(), v.len() as u64, 99) }, 0);
    }

    #[test]
    fn opening_a_missing_library_errors_instead_of_panicking() {
        let e = LoadedLibrary::open("no_such_plugin.dll").unwrap_err();
        assert!(
            matches!(e, MtbError::Other(_) | MtbError::Io(_)),
            "应为加载失败: {e:?}"
        );
    }

    #[test]
    fn null_symbol_maps_to_3102() {
        let e: MtbError =
            unsafe { resolve::<extern "C" fn() -> u32>(std::ptr::null_mut(), "mb_nope") }.unwrap_err();
        match e {
            MtbError::Coded { code, .. } => assert_eq!(code, MtbError::PLUGIN_SYM_MISSING),
            other => panic!("期望 3102，得到 {other:?}"),
        }
    }

    #[test]
    fn shape_helpers_roundtrip() {
        let shape = vec![2usize, 3, 4];
        let d = dims_u32(&shape);
        assert_eq!(d, vec![2, 3, 4]);
        assert_eq!(shape_of(&d), shape);
        // 视图的 dims 内联拷贝，秩超过上限会被截断，因此调用方必须先校验
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let v = HostTensor::view(&t.data, &dims_u32(&t.shape));
        assert_eq!(v.len, 4);
        assert_eq!(v.dims[..2], [2, 2]);
    }
}
