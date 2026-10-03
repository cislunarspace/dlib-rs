//! `Array2D<T>` row-major 2D array and the `GenericImage` trait, ported from
//! dlib's `dlib/array2d/array2d_kernel.h` (data layout, `set_size`,
//! serialization) and `dlib/matrix/matrix_generic_image.h` +
//! `dlib/array2d/array2d_generic_image.h` (the `generic_image` interface).

use std::ops::{Index, IndexMut};

use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::serialize::{Deserializer, DlibSerialize, SerializeError, Serializer};

/// Row-major 2D array, port of `dlib::array2d<T>` (`dlib/array2d/array2d_kernel.h`).
///
/// Divergence from C++: `set_size` and `zeros` default-fill the new elements,
/// whereas the C++ version leaves them uninitialized.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct Array2D<T> {
    data: Vec<T>,
    nr: usize,
    nc: usize,
}

impl<T: Clone + Default> Array2D<T> {
    /// Creates an empty array (`array2d()` in C++).
    pub fn new() -> Self {
        Array2D {
            data: Vec::new(),
            nr: 0,
            nc: 0,
        }
    }

    /// Creates an `nr x nc` array of default values.
    pub fn zeros(nr: usize, nc: usize) -> Self {
        Array2D {
            data: vec![T::default(); nr * nc],
            nr,
            nc,
        }
    }

    /// Resizes to `nr x nc` (`array2d::set_size`).
    ///
    /// Unlike C++, existing logical elements are not preserved on shrink and
    /// new elements are default-filled rather than left uninitialized.
    pub fn set_size(&mut self, nr: usize, nc: usize) {
        self.data.clear();
        self.data.resize(nr * nc, T::default());
        self.nr = nr;
        self.nc = nc;
    }

    /// Number of rows.
    pub fn nr(&self) -> usize {
        self.nr
    }

    /// Number of columns.
    pub fn nc(&self) -> usize {
        self.nc
    }

    /// `nr * nc`.
    pub fn size(&self) -> usize {
        self.nr * self.nc
    }

    /// Element at `(r, c)` (`array2d[r][c]` in C++). Panics out of bounds.
    pub fn get(&self, r: usize, c: usize) -> &T {
        &self.data[r * self.nc + c]
    }

    /// Mutable element at `(r, c)`.
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
        &mut self.data[r * self.nc + c]
    }

    /// Row `r` as a slice (`array2d[r]` in C++).
    pub fn row(&self, r: usize) -> &[T] {
        &self.data[r * self.nc..(r + 1) * self.nc]
    }

    /// Mutable row `r`.
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        &mut self.data[r * self.nc..(r + 1) * self.nc]
    }

    /// Iterates all elements in row-major order (`array2d` enumerator).
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Serializes exactly like `serialize(const array2d<T>&)` in
    /// `dlib/array2d/array2d_kernel.h`: packed signed `-(nr as i64)`,
    /// `-(nc as i64)`, then the elements in row-major order (the negated
    /// dims keep the format compatible with dlib's matrix serialization).
    pub fn serialize(&self, out: &mut Serializer)
    where
        T: DlibSerialize,
    {
        out.write_i64(-(self.nr as i64));
        out.write_i64(-(self.nc as i64));
        for x in &self.data {
            x.dlib_serialize(out);
        }
    }

    /// Deserializes exactly like `deserialize(array2d<T>&)` in
    /// `dlib/array2d/array2d_kernel.h`: two packed i64 dims; if either is
    /// negative both are negated (current format), otherwise the dims are
    /// swapped (legacy pre-negation format); then `nr*nc` row-major elements.
    pub fn deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError>
    where
        T: DlibSerialize,
    {
        let mut nr = inp.read_i64()?;
        let mut nc = inp.read_i64()?;
        if nr < 0 || nc < 0 {
            nr = -nr;
            nc = -nc;
        } else {
            std::mem::swap(&mut nr, &mut nc);
        }
        if nr < 0 || nc < 0 {
            return Err(SerializeError::Malformed("array2d dims"));
        }
        let count = nr
            .checked_mul(nc)
            .ok_or(SerializeError::Malformed("array2d dims"))?;
        // Every element occupies at least one byte, so a count larger than
        // the remaining stream can never be satisfied.
        if count as u128 > inp.remaining() as u128 {
            return Err(SerializeError::Eof);
        }
        let mut data = Vec::with_capacity(count.min(1 << 20) as usize);
        for _ in 0..count {
            data.push(T::dlib_deserialize(inp)?);
        }
        Ok(Array2D {
            data,
            nr: nr as usize,
            nc: nc as usize,
        })
    }
}

impl<T: Clone + Default> Index<(usize, usize)> for Array2D<T> {
    type Output = T;
    fn index(&self, (r, c): (usize, usize)) -> &T {
        self.get(r, c)
    }
}

impl<T: Clone + Default> IndexMut<(usize, usize)> for Array2D<T> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        self.get_mut(r, c)
    }
}

impl<T: Clone + Default> Index<usize> for Array2D<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Clone + Default> IndexMut<usize> for Array2D<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<T: Clone + Default + DlibSerialize> DlibSerialize for Array2D<T> {
    fn dlib_serialize(&self, out: &mut Serializer) {
        self.serialize(out);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        Array2D::deserialize(inp)
    }
}

// ----------------------------------------------------------------------------------------

/// Trait for 2D image types, port of the `generic_image` interface
/// (`dlib/matrix/matrix_generic_image.h`, `image_view`/`const_image_view`
/// accessors; implemented for `array2d` in
/// `dlib/array2d/array2d_generic_image.h`).
pub trait GenericImage {
    /// Pixel type of the image (`pixel_traits` basic type).
    type PixelType;

    /// Number of rows (`num_rows`).
    fn num_rows(&self) -> usize;
    /// Number of columns (`num_columns`).
    fn num_columns(&self) -> usize;
    /// Resizes the image (`set_image_size`).
    fn set_image_size(&mut self, rows: usize, cols: usize);
    /// Immutable pixel access (`image_data()[r][c]`).
    fn pixel(&self, r: usize, c: usize) -> &Self::PixelType;
    /// Mutable pixel access.
    fn pixel_mut(&mut self, r: usize, c: usize) -> &mut Self::PixelType;
}

impl<T: Clone + Default> GenericImage for Array2D<T> {
    type PixelType = T;

    fn num_rows(&self) -> usize {
        self.nr()
    }
    fn num_columns(&self) -> usize {
        self.nc()
    }
    fn set_image_size(&mut self, rows: usize, cols: usize) {
        self.set_size(rows, cols);
    }
    fn pixel(&self, r: usize, c: usize) -> &T {
        self.get(r, c)
    }
    fn pixel_mut(&mut self, r: usize, c: usize) -> &mut T {
        self.get_mut(r, c)
    }
}

impl<T: Clone + Default> GenericImage for Matrix<T> {
    type PixelType = T;

    fn num_rows(&self) -> usize {
        self.nr()
    }
    fn num_columns(&self) -> usize {
        self.nc()
    }
    fn set_image_size(&mut self, rows: usize, cols: usize) {
        self.set_size(rows, cols);
    }
    fn pixel(&self, r: usize, c: usize) -> &T {
        &self[(r, c)]
    }
    fn pixel_mut(&mut self, r: usize, c: usize) -> &mut T {
        &mut self[(r, c)]
    }
}

// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_and_rows() {
        let mut a = Array2D::<u32>::zeros(3, 4);
        assert_eq!(a.nr(), 3);
        assert_eq!(a.nc(), 4);
        assert_eq!(a.size(), 12);
        for r in 0..3 {
            for c in 0..4 {
                a[(r, c)] = (r * 4 + c) as u32;
            }
        }
        assert_eq!(a[(1, 2)], 6);
        assert_eq!(a[6], 6); // flat row-major index
        assert_eq!(*a.get(2, 0), 8);
        *a.get_mut(0, 0) = 99;
        assert_eq!(a[(0, 0)], 99);
        assert_eq!(a.row(1), &[4, 5, 6, 7]);
        a.row_mut(2)[3] = 77;
        assert_eq!(a[(2, 3)], 77);
        assert_eq!(
            a.iter().copied().sum::<u32>(),
            99 + (1..=11).sum::<u32>() - 11 + 77
        );
        // set_size default-fills.
        a.set_size(2, 2);
        assert_eq!(a.nr(), 2);
        assert_eq!(a.size(), 4);
        assert!(a.iter().all(|&x| x == 0));
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds_panics() {
        let a = Array2D::<u8>::zeros(2, 2);
        let _ = a.get(2, 0);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut a = Array2D::<u8>::zeros(3, 4);
        for i in 0..12 {
            a[i] = i as u8;
        }
        let mut ser = Serializer::new();
        a.serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        let b = Array2D::<u8>::deserialize(&mut de).unwrap();
        assert_eq!(a, b);
        assert_eq!(de.remaining(), 0);

        // f64 elements and DlibSerialize delegation.
        let mut m = Array2D::<f64>::zeros(2, 3);
        m[(0, 0)] = -1.5;
        m[(1, 2)] = 2.25;
        let mut ser = Serializer::new();
        m.dlib_serialize(&mut ser);
        let bytes = ser.into_inner();
        let back = Array2D::<f64>::dlib_deserialize(&mut Deserializer::new(&bytes)).unwrap();
        assert_eq!(m, back);

        // Truncated stream errors instead of panicking.
        assert!(Array2D::<u8>::deserialize(&mut Deserializer::new(&bytes[..2])).is_err());
        // Legacy positive-dims format swaps nr/nc (dlib deserialize).
        let mut ser = Serializer::new();
        ser.write_i64(3); // positive => legacy: swap -> nr=4? no: swap(nr,nc) => nr=4? see below
        ser.write_i64(4);
        for v in 0u8..12 {
            ser.write_u8(v);
        }
        let back = Array2D::<u8>::deserialize(&mut Deserializer::new(&ser.into_inner())).unwrap();
        assert_eq!(back.nr(), 4);
        assert_eq!(back.nc(), 3);
        assert_eq!(back[(0, 0)], 0);
        assert_eq!(back[(3, 2)], 11);
    }

    #[test]
    fn test_generic_image_for_matrix_and_array2d() {
        let mut a = Array2D::<u8>::zeros(2, 2);
        *a.pixel_mut(0, 1) = 5;
        assert_eq!(*a.pixel(0, 1), 5);
        a.set_image_size(3, 1);
        assert_eq!(a.num_rows(), 3);
        assert_eq!(a.num_columns(), 1);

        let mut m = Matrix::<f64>::zeros(2, 3);
        assert_eq!(m.num_rows(), 2);
        assert_eq!(m.num_columns(), 3);
        *m.pixel_mut(1, 2) = 7.0;
        assert_eq!(*m.pixel(1, 2), 7.0);
        m.set_image_size(4, 4);
        assert_eq!(m.num_rows(), 4);
        assert_eq!(m.num_columns(), 4);
        assert_eq!(*m.pixel(0, 0), 0.0);
    }
}
