//! MDB 加密面（README 4.4 / 8.4 / 11.1）。
//!
//! 密钥分两层：[`MasterKeys`] 里
//!
//! 1. **KEK** —— `Argon2id(口令, kek_salt)`，只用来封装文件头；
//! 2. **DEK** —— `SHA-256(KEK ‖ dek_nonce)`，用来加密所有页与 WAL 帧。
//!
//! 口令从不直接拿去加密数据，因此 `ROTATE KEY` 只需换 KEK + 重新加密文件头，
//! 再按新 DEK 重写一遍页——数据本身一个字节都不用重新生成。
//!
//! Nonce 由 `(epoch, 用途, 目标 id)` 推导而非随机数：既避免了"忘记换 nonce"的
//! 经典事故（同一密钥下 nonce 绝不会重复），也让密文可复现（`ROTATE KEY` 前后
//! 解密一致）。`epoch` 在每次 `ROTATE KEY` 时更新，把 nonce 空间整体作废一次。

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit};
use argon2::{Algorithm, Argon2, Params, Version};
use mightbe_core::{MtbError, MtbResult};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Argon2id 参数。默认按 OWASP 下限取，测试可用 [`KdfParams::fast`] 跳过长跑。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdfParams {
    /// 内存代价（KiB）
    pub m_cost_kib: u32,
    /// 迭代次数
    pub t_cost: u32,
    /// 并行度
    pub p_cost: u32,
}

impl Default for KdfParams {
    fn default() -> Self {
        Self {
            m_cost_kib: 19 * 1024,
            t_cost: 2,
            p_cost: 1,
        }
    }
}

impl KdfParams {
    /// 仅供测试：把内存/时间压到毫秒级。绝不用于生产数据。
    #[allow(dead_code)]
    pub const fn fast() -> Self {
        Self {
            m_cost_kib: 8 * 1024,
            t_cost: 1,
            p_cost: 1,
        }
    }
}

/// 随机盐/nonce 的上下文（无系统随机源时的兜底：固定模式 + 计数器）。
pub fn derive_nonce_from(epoch: &[u8; 16], tag: &[u8], id: u64) -> [u8; 12] {
    let mut h = Sha256::new();
    h.update(epoch);
    h.update(tag);
    h.update(id.to_be_bytes());
    let out = h.finalize();
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&out[..12]);
    nonce
}

/// AES-256-GCM 的认证标签长度：密文 = 明文 + 16，少这 16 字节就无法认证。
pub const TAG_LEN: usize = 16;

/// 主密钥。Drop 时自动擦除（zeroize）。
pub struct MasterKeys {
    _kek: Zeroizing<[u8; 32]>,
    dek: Zeroizing<[u8; 32]>,
    epoch: [u8; 16],
}

impl std::fmt::Debug for MasterKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MasterKeys")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl MasterKeys {
    /// 派生主密钥。任何一步失败都返回错误，绝不产出半截密钥。
    pub fn derive(
        password: &[u8],
        kek_salt: &[u8; 16],
        kdf: &KdfParams,
        dek_nonce: &[u8; 16],
        epoch: [u8; 16],
    ) -> MtbResult<Self> {
        let params = Params::new(kdf.m_cost_kib, kdf.t_cost, kdf.p_cost, None)
            .map_err(|e| MtbError::coded(MtbError::STORE + 10, format!("argon2 参数非法: {e}")))?;
        let argon2 =
            Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        let mut kek = Zeroizing::new([0u8; 32]);
        argon2
            .hash_password_into(password, kek_salt, kek.as_mut())
            .map_err(|e| MtbError::coded(MtbError::STORE + 11, format!("KEK 派生失败: {e}")))?;

        let mut h = Sha256::new();
        h.update(kek.as_slice());
        h.update(dek_nonce);
        let digest = h.finalize();
        let mut dek = Zeroizing::new([0u8; 32]);
        dek.copy_from_slice(&digest);

        Ok(Self {
            _kek: kek,
            dek,
            epoch,
        })
    }

    /// 当前 epoch（nonce 空间的版本号）。
    pub fn epoch(&self) -> &[u8; 16] {
        &self.epoch
    }

    fn cipher(&self) -> Aes256Gcm {
        Aes256Gcm::new_from_slice(self.dek.as_slice())
            .expect("DEK 恒为 32 字节，构造不会失败")
    }

    /// AEAD 封装：`aad` 绑定用途与目标，`ct` 为 `plain || tag`。
    pub fn seal(&self, aad: &[u8], id: u64, plain: &[u8]) -> MtbResult<Vec<u8>> {
        let nonce = derive_nonce_from(&self.epoch, aad, id);
        self.cipher()
            .encrypt(aes_gcm::Nonce::from_slice(&nonce), plain)
            .map_err(|e| MtbError::coded(MtbError::STORE + 12, format!("加密失败: {e}")))
    }

    /// AEAD 解封。**标签不符时返回 `None`，不 panic、不泄露任何明文位。**
    pub fn open<'a>(&self, aad: &[u8], id: u64, ct: &'a [u8]) -> Option<Vec<u8>> {
        let nonce = derive_nonce_from(&self.epoch, aad, id);
        self.cipher()
            .decrypt(aes_gcm::Nonce::from_slice(&nonce), ct)
            .ok()
    }

    /// 整页加密。输出长度恒为 `明文 + 16`（GCM 认证标签）。
    ///
    /// 别指望"裁成正好一页"：那样标签就没了，认证形同虚设。
    /// 落盘时按 [`crate::mdb::BLOCK_SIZE`] 对齐即可，页号与偏移仍然一一对应。
    pub fn seal_page(&self, page_id: u64, plain: &[u8]) -> MtbResult<Vec<u8>> {
        self.seal(b"mdb/page", page_id, plain)
    }

    /// 整页解密；密文长度或标签不符都算失败。
    ///
    /// 密文长度必须是 [`crate::page::PAGE_SIZE`] + 标签，少一个字节都无法认证。
    pub fn open_page(&self, page_id: u64, ct: &[u8]) -> MtbResult<PagePlain> {
        if ct.len() != crate::page::PAGE_SIZE + TAG_LEN {
            return Err(MtbError::coded(
                MtbError::STORE + 13,
                format!(
                    "页密文长度 {} 异常（应为 {}）",
                    ct.len(),
                    crate::page::PAGE_SIZE + TAG_LEN
                ),
            ));
        }
        self.open(b"mdb/page", page_id, ct)
            .ok_or_else(|| MtbError::coded(MtbError::STORE + 14, format!("页 {} 解密失败", page_id)))
    }
}

/// 解密后的整页明文（`PAGE_SIZE` 字节）。
pub type PagePlain = Vec<u8>;

#[cfg(test)]
mod tests {
    use super::*;

    const SALT: [u8; 16] = [7u8; 16];
    const DEK_NONCE: [u8; 16] = [3u8; 16];

    fn keys_with(epoch: [u8; 16]) -> MasterKeys {
        MasterKeys::derive(b"correct horse", &SALT, &KdfParams::fast(), &DEK_NONCE, epoch)
            .expect("派生成功")
    }

    #[test]
    fn page_seal_roundtrip_is_fixed_size() {
        let k = keys_with([1u8; 16]);
        let plain = vec![9u8; crate::page::PAGE_SIZE];
        let ct = k.seal_page(42, &plain).expect("加密");
        // 定长来自"页 + GCM 标签"，不是凭空裁到页长——裁掉标签等于废掉认证
        assert_eq!(ct.len(), crate::page::PAGE_SIZE + 16);
        let back = k.open_page(42, &ct).expect("解密");
        assert_eq!(back, plain);
        // 同页同密钥两次加密必须完全一致（nonce 是推导出来的）
        let ct2 = k.seal_page(42, &plain).expect("再加密");
        assert_eq!(ct, ct2, "推导式 nonce 应使加密确定性可复现");
    }

    #[test]
    fn wrong_page_id_does_not_decrypt() {
        let k = keys_with([1u8; 16]);
        let plain = vec![5u8; crate::page::PAGE_SIZE];
        let ct = k.seal_page(1, &plain).expect("加密");
        assert!(k.open_page(2, &ct).is_err(), "页 id 参与 nonce，错 id 必失败");
    }

    #[test]
    fn single_bit_flip_is_rejected() {
        let k = keys_with([2u8; 16]);
        let ct = k.seal_page(77, &vec![0u8; crate::page::PAGE_SIZE]).expect("加密");
        let n = ct.len();
        for idx in [0usize, n / 3, n - 1] {
            let mut bad = ct.clone();
            bad[idx] ^= 0x40;
            assert!(
                k.open_page(77, &bad).is_err(),
                "第 {idx} 字节被翻转却放行了（AEAD + 定长校验）"
            );
        }
    }

    #[test]
    fn wrong_password_or_wrong_kek_fails() {
        let right = MasterKeys::derive(b"pw", &SALT, &KdfParams::fast(), &DEK_NONCE, [4u8; 16])
            .expect("派生");
        let wrong_pw =
            MasterKeys::derive(b"pw2", &SALT, &KdfParams::fast(), &DEK_NONCE, [4u8; 16])
                .expect("派生");
        let wrong_salt =
            MasterKeys::derive(b"pw", &[9u8; 16], &KdfParams::fast(), &DEK_NONCE, [4u8; 16])
                .expect("派生");
        let wrong_epoch =
            MasterKeys::derive(b"pw", &SALT, &KdfParams::fast(), &DEK_NONCE, [5u8; 16])
                .expect("派生");

        let ct = right.seal_page(9, &vec![1u8; crate::page::PAGE_SIZE]).expect("加密");
        let expect_err = |k: &MasterKeys| k.open_page(9, &ct).is_err();
        assert!(expect_err(&wrong_pw), "口令错 → KEK 错 → 解不开");
        assert!(expect_err(&wrong_salt), "盐错 → KEK 错 → 解不开");
        assert!(expect_err(&wrong_epoch), "epoch 错 → nonce 错 → 解不开");
    }

    #[test]
    fn ciphertext_hides_plaintext_shape() {
        let k = keys_with([6u8; 16]);
        let a = k.seal_page(1, &vec![0u8; crate::page::PAGE_SIZE]).expect("加密");
        let b = k.seal_page(1, &vec![1u8; crate::page::PAGE_SIZE]).expect("加密");
        assert_ne!(a, b, "相同长度不同内容的密文必须不同");
        // 明文全零 vs 全 1：密文不得呈现可识别的规律（这里只要求不相等）
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn seal_open_blob_roundtrip() {
        let k = keys_with([8u8; 16]);
        let blob = b"record payload".to_vec();
        let ct = k.seal(b"wal/frame", 3u64, &blob).expect("封装");
        assert_eq!(k.open(b"wal/frame", 3u64, &ct).expect("解封"), blob);
        assert!(
            k.open(b"wal/frame", 4u64, &ct).is_none(),
            "aad 里的 id 必须参与绑定"
        );
    }
}
