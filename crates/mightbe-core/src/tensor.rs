//! 行主序多维张量（README 4.1）。
//!
//! MVP 全部 CPU f32、行主序；`strides` 由构造函数统一推导，
//! 保证 `.reshape()` / `.transpose2d()` 之后索引语义不变。

use crate::api::{MtbError, MtbResult, Shape};

/// 计算行主序 strides。`[a,b,c] -> [b*c, c, 1]`。
pub fn strides_of(shape: &[usize]) -> Shape {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for (i, &d) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc *= d.max(1);
    }
    strides
}

/// 元素总数；空形状视为 1（标量）。
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Shape,
    pub strides: Shape,
}

impl Tensor {
    pub fn zeros(shape: Shape) -> Self {
        let strides = strides_of(&shape);
        let n = numel(&shape);
        Self { data: vec![0.0; n], strides, shape }
    }

    /// 由已有数据构造，长度必须等于 `shape` 的元素数，否则形状错。
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> MtbResult<Self> {
        if data.len() != numel(&shape) {
            return Err(MtbError::Shape {
                expected: format!("{} elements", numel(&shape)),
                got: format!("{} elements", data.len()),
            });
        }
        let strides = strides_of(&shape);
        Ok(Self { data, shape, strides })
    }

    /// 全 1 张量（偏置、Gate 初始化常用）。
    pub fn ones(shape: Shape) -> Self {
        let mut t = Self::zeros(shape);
        for x in t.data.iter_mut() {
            *x = 1.0;
        }
        t
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 标量取值（形状必须为空或 [1]）。
    pub fn scalar(&self) -> MtbResult<f32> {
        if self.data.len() == 1 {
            Ok(self.data[0])
        } else {
            Err(MtbError::Shape {
                expected: "scalar (0-d or 1 element)".into(),
                got: format!("{} elements", self.data.len()),
            })
        }
    }

    /// 多下标取值。越界返回形状错误而非 panic。
    pub fn get(&self, idx: &[usize]) -> MtbResult<f32> {
        if idx.len() != self.shape.len() {
            return Err(MtbError::Shape {
                expected: format!("{} dims", self.shape.len()),
                got: format!("{} dims", idx.len()),
            });
        }
        let flat = self.offset(idx);
        self.data
            .get(flat)
            .copied()
            .ok_or_else(|| MtbError::Shape {
                expected: format!("offset < {}", self.data.len()),
                got: format!("offset {flat}"),
            })
    }

    /// 多下标写入。
    pub fn set(&mut self, idx: &[usize], v: f32) -> MtbResult<()> {
        let flat = self.offset(idx);
        match self.data.get_mut(flat) {
            Some(slot) => {
                *slot = v;
                Ok(())
            }
            None => Err(MtbError::Shape {
                expected: format!("offset < {}", self.data.len()),
                got: format!("offset {flat}"),
            }),
        }
    }

    fn offset(&self, idx: &[usize]) -> usize {
        idx.iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }

    /// 改变形状（总元素数不变），否则形状错。
    pub fn reshape(&self, shape: Shape) -> MtbResult<Tensor> {
        if numel(&shape) != self.data.len() {
            return Err(MtbError::Shape {
                expected: format!("{} elements", self.data.len()),
                got: format!("{} elements", numel(&shape)),
            });
        }
        let strides = strides_of(&shape);
        Ok(Tensor { data: self.data.clone(), shape, strides })
    }

    /// 二维转置（只支持 ndim==2；RNN/矩阵算子够用）。
    pub fn transpose2d(&self) -> MtbResult<Tensor> {
        if self.shape.len() != 2 {
            return Err(MtbError::Shape {
                expected: "2-d tensor".into(),
                got: format!("{}d", self.shape.len()),
            });
        }
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut data = vec![0f32; self.data.len()];
        for i in 0..r {
            for j in 0..c {
                data[j * r + i] = self.data[i * c + j];
            }
        }
        Ok(Tensor {
            data,
            shape: vec![c, r],
            strides: strides_of(&[c, r]),
        })
    }

    /// 与另一个同形张量逐元素相加（原地，避免训练循环里频繁分配）。
    pub fn add_inplace(&mut self, other: &Tensor) -> MtbResult<()> {
        if self.shape != other.shape {
            return Err(MtbError::Shape {
                expected: format!("{:?}", self.shape),
                got: format!("{:?}", other.shape),
            });
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
        Ok(())
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Tensor::zeros(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides_of(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn reshape_ok_and_bad() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.reshape(vec![4]).unwrap().shape, vec![4]);
        assert!(t.reshape(vec![3]).is_err());
    }

    #[test]
    fn index_roundtrip_after_transpose() {
        let t = Tensor::from_vec((0..6).map(|i| i as f32).collect::<Vec<f32>>(), vec![2, 3]).unwrap();
        let tt = t.transpose2d().unwrap();
        assert_eq!(tt.get(&[2, 1]).unwrap(), t.get(&[1, 2]).unwrap());
    }

}
