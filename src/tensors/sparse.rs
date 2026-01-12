//! Sparse linear algebra methods using matrices and vectors.
//!
//! # Example
//!
//! ```rust
//! use numerica::tensors::sparse::{SparseMatrix};
//! use numerica::domains::{
//! 	integer::{IntegerRing, Integer},
//!     rational::{FractionField}
//! };
//! let r = IntegerRing::new();
//!
//! let mat = SparseMatrix::from_csr(2,3,vec![Integer::new(10),Integer::new(7),Integer::new(13)],vec![0,2,3],vec![0,2,1],r);
//! assert_eq!(mat.fmt_mma(), "{{{1,1}->10,{1,3}->7,{2,2}->13},{2,3}}");

use std::{
    ops::{Neg, Add, AddAssign, Sub, SubAssign, Mul, MulAssign},
    collections::{HashSet}
};

use itertools::Itertools;

use rand::Rng;

use rayon::prelude::*;

use crate::{
    domains::{
        Ring, Field, RingPrinter,
        integer::{IntegerRing, Integer},
        rational::{FractionField, Fraction}
    }
};

/// A sparse vector in a compressed format (compressed sparse row style).
///
/// We keep the entries sorted by their index at all times.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SparseVector<F: Ring> {
    /// The non-zero entries of the vector sorted by index.
    pub(crate) values: Vec<F::Element>,
    /// The indices corresponding to the entries of `values`.
    /// Must have the same length a `values`.
    pub(crate) idcs: Vec<u32>,
    /// The size/length of the vector.
    pub(crate) len : u32,
    /// The ring/field of the elements of the matrix
    pub(crate) field: F,
}

impl<F: Ring> SparseVector<F> {
    /// Create a new zeroed sparse vector over the ring/feld `F` of length `len`.
    pub fn new(len : u32, field: F) -> SparseVector<F> {
        SparseVector {
            values : Vec::new(),
            idcs : Vec::new(),
            len : len,
            field : field
        }
    }

    /// Create a new sparse vector from CSR-like data.
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `values` - Sorted (by index) non-zero entries of the vector..
    /// * `idcs` - The sorted indices/positions of the values.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_csr(len : u32, values : Vec<F::Element>, idcs : Vec<u32>, field : F) -> SparseVector<F> {
        SparseVector {
            values,
            idcs,
            len,
            field
        }
    }

    /// Create a new sparse vector from CSR-like data.
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `values` - Sorted (by index) non-zero entries of the vector..
    /// * `idcs` - The sorted indices/positions of the values.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_csr_slices(len : u32, values : &[F::Element], idcs : &[u32], field : F) -> SparseVector<F> {
        SparseVector {
            values : values.to_vec(),
            idcs : idcs.to_vec(),
            len,
            field
        }
    }

    /// Create a new sparse vector from (pos, value) pairs.
    ///
    /// The values should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `pairs` - (pos, vaule) pairs of the non-zero entries.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_pairs(len : u32, pairs : Vec<(u32, F::Element)>, field : F) -> SparseVector<F> {
        let n = pairs.len();
        let mut idcs = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for (i, v) in pairs {
            idcs.push(i);
            values.push(v);
        }
        
        SparseVector {
            values,
            idcs,
            len,
            field
        }
    }

    /// Create a new sparse vector from (pos, value) pairs.
    ///
    /// The values should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `pairs` - (pos, vaule) pairs of the non-zero entries.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_pairs_slices(len : u32, pairs : &[(u32, F::Element)], field : F) -> SparseVector<F> {
        let n = pairs.len();
        let mut idcs = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for (i, v) in pairs {
            idcs.push(*i);
            values.push(v.clone());
        }
        
        SparseVector {
            values,
            idcs,
            len,
            field
        }
    }

    /// Return the length of the vector .
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Format in Mathematica form.
    ///
    /// Simply apply `SparseArray@@` to the output in MMA.
    pub fn fmt_mma(&self) -> String {
        let vals = self.idcs.iter().zip(self.values.iter()).map(|(idx, val)| {
            //format each element as {idx,col}->val
            let val_printer = RingPrinter::new(&self.field, &val);
            format!("{{{},{}}}->{}", idx + 1, 1, val_printer)
        }).join(",");

        format!("{{{{{}}},{{{},{}}}}}", vals, self.len, 1)
    }
}

/// An error enum for some functions of SparseMatrix.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum SparseMatrixError<F: Ring> {
    /// Shape of matrix and vector don't match
    ShapeMismatch,
    /// Fields of arguments don't match
    FieldMismatch,
    /// No solution exists
    Inconsistent,
    /// System is underdetermined (infinite solutions)
    Underdetermined {
        rank: usize,
        row_reduced_augmented_matrix: SparseMatrix<F>,
    },
}

/// A sparse matrix in compressed sparse row (CSR) format.
///
/// We keep each row sorted at all times.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SparseMatrix<F: Ring> {
    /// The non-zero entries of the matrix sorted by row and column
    pub(crate) values: Vec<F::Element>,
    /// Indices where new rows start within `values`, including an after-end index.
    /// Has length nrows + 1.
    pub(crate) row_idcs: Vec<usize>,
    /// The column indices corresponding to the entries of `values`.
    /// Must have the same length a `values`.
    pub(crate) col_idcs: Vec<u32>,
    /// Number of rows
    pub(crate) nrows: u32,
    /// Number of columns
    pub(crate) ncols: u32,
    /// The ring/field of the elements of the matrix
    pub(crate) field: F,
}

impl<F: Ring> SparseMatrix<F> {
    /// Create a new zeroed sparse matrix over the ring/field `F` with `nrows` rows and `ncols` columns
    ///
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `field` - the field of the matrix entries
    pub fn new(nrows: u32, ncols: u32, field: F) -> SparseMatrix<F> {
        SparseMatrix {
            values: Vec::new(),
            row_idcs: vec![0; (nrows + 1) as usize],
            col_idcs: Vec::new(),
            nrows,
            ncols,
            field
        }
    }

    /// Create a new sparse matrix over the ring/field `F` from explicit CSR data
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `values` - non-zero entries sorted by row and column
    /// * `row_idcs` - indices where new rows start within `values`, including an after-end index
    /// * `col_idcs` - column indices corresponding to entries of `values`
    /// * `field` - the field of the matrix entries
    pub fn from_csr(nrows: u32, ncols: u32, values: Vec<F::Element>, row_idcs: Vec<usize>, col_idcs: Vec<u32>, field: F) -> SparseMatrix<F> {
        assert!(values.len() == col_idcs.len());
        assert!(row_idcs.len() == ((nrows + 1) as usize));
        SparseMatrix {
            values,
            row_idcs,
            col_idcs,
            nrows,
            ncols,
            field
        }
    }

    /// Create a new sparse matrix over the ring/field `F` from explicit CSR data
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `values` - non-zero entries sorted by row and column
    /// * `row_idcs` - indices where new rows start within `values`, including an after-end index
    /// * `col_idcs` - column indices corresponding to entries of `values`
    /// * `field` - the field of the matrix entries
    pub fn from_csr_slices(nrows: u32, ncols: u32, values: &[F::Element], row_idcs: &[usize], col_idcs: &[u32], field: F) -> SparseMatrix<F> {
        assert!(values.len() == col_idcs.len());
        assert!(row_idcs.len() == ((nrows + 1) as usize));
        SparseMatrix {
            values : values.to_vec(),
            row_idcs : row_idcs.to_vec(),
            col_idcs : col_idcs.to_vec(),
            nrows,
            ncols,
            field
        }
    }

    /// Create a sparse matrix from ordered triplets of (row, column, entry)
    ///
    /// # Arguments
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `triplets` - ordered(!) triplets of (row, column, entry). Row and column indices are 0-indexed
    /// * `field` - the ring/field of the matrix entries
    ///
    /// # Example
    /// ```rust
    /// use numerica::tensors::sparse::{SparseMatrix};
    /// use numerica::domains::integer::{IntegerRing, Integer};
    /// let r = IntegerRing::new();
    ///
    /// let mat = SparseMatrix::from_triplets(4,3, vec![(0,0,Integer::new(15)),(0,2,Integer::new(-23)),(2,1,Integer::new(-7)),(2,2,Integer::new(2)),(3,0,Integer::new(-1))], r);
    /// println!("{}", mat.fmt_mma());
    /// assert_eq!(mat.fmt_mma(), "{{{1,1}->15,{1,3}->-23,{3,2}->-7,{3,3}->2,{4,1}->-1},{4,3}}");
    /// ```
    /// 
    pub fn from_triplets(nrows: u32, ncols: u32, triplets : Vec<(u32, u32, F::Element)>, field: F) -> SparseMatrix<F> {
        debug_assert!(triplets.is_sorted_by_key(|&(row, col, _) | (row, col)));
        let mut ret = SparseMatrix {
            values: Vec::with_capacity(triplets.len()),
            row_idcs: Vec::with_capacity((nrows + 1) as usize),
            col_idcs: Vec::with_capacity(triplets.len()),
            nrows,
            ncols,
            field,
        };
        ret.row_idcs.push(0);
        let mut current_row : u32 = 0;
        for (row, col, el) in triplets {
            while current_row < row {
                //start new row/insert empty rows
                ret.row_idcs.push(ret.values.len());
                current_row += 1;
            }
            ret.values.push(el);
            ret.col_idcs.push(col);
        }
        //finish up the row_idcs
        while current_row < ret.nrows {
            ret.row_idcs.push(ret.values.len());
            current_row += 1;
        }
        debug_assert!(ret.row_idcs.len() == (ret.nrows + 1) as usize);

        ret
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> u32 {
        self.nrows as u32
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> u32 {
        self.ncols as u32
    }

    /// Return the field of the matrix entries.
    pub fn field(&self) -> &F {
        &self.field
    }

    /// Return the number of non-zero entries.
    pub fn nvalues(&self) -> usize {
        self.values.len()
    }

    /// Multiply the scalar `e` to each entry of the matrix
    pub fn mul_scalar(&self, el: &F::Element) -> SparseMatrix<F> {
        let mut ret = SparseMatrix {
            values: self.values.iter().map(|ell| self.field.mul(ell, el)).collect(),
            row_idcs: self.row_idcs.clone(),
            col_idcs: self.col_idcs.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone()
        };
        ret.erase_zeroes();

        ret
    }

    /// Add a new row to the matrix
    ///
    /// # Parameters
    /// * `values` - the values of the non-zero entries of the row
    /// * `col_idcs` - the column indices of the non-zero entries of the row
    pub fn add_row(&mut self, values : Vec<F::Element>, col_idcs : Vec<u32>) -> () {
        self.row_idcs.push(self.row_idcs.last().unwrap() + values.len());
        self.values.extend(values);
        self.col_idcs.extend(col_idcs);
        self.nrows += 1;
    }

    /// Add empty columns to the matrix.
    ///
    /// * `col_pos` - Ordered(!) positions where the new columns should be inserted. Each entry must NOT account for previously inserted columns.
    pub fn add_cols(&mut self, col_pos : &Vec<u32>) -> () {
        debug_assert!(col_pos.is_sorted());

        //update ncols
        self.ncols += col_pos.len() as u32;

        //update col_idcs, row by row
        for pair in self.row_idcs.windows(2) {
            let mut col_pos_it : u32 = 0;
            for pos in pair[0]..pair[1] {
                //advance iterator as long *it <= col_idx at pos
                while (col_pos_it as usize) < col_pos.len() && col_pos[col_pos_it as usize] <= self.col_idcs[pos] {
                    col_pos_it += 1;
                }
                //shift the current col_idx by the number of new columns that are inserted before it
                self.col_idcs[pos] += col_pos_it;
            }
        }
    }

    /// Append a column to the right of the matrix
    pub fn append_col(&mut self, col: SparseVector<F>) -> () {
        debug_assert_eq!(col.values.len(), col.idcs.len());
        assert_eq!(col.len(), self.nrows);

        let old_values = std::mem::take(&mut self.values);
        let old_col_idcs = std::mem::take(&mut self.col_idcs);

        let mut new_values = Vec::with_capacity(old_values.len() + col.values.len());
        let mut new_col_idcs = Vec::with_capacity(old_col_idcs.len() + col.idcs.len());

        let mut old_values_iter = old_values.into_iter();
        let mut old_col_idcs_iter = old_col_idcs.into_iter();
        let mut col_iter = col.idcs.into_iter().zip(col.values.into_iter());

        let mut current_col = col_iter.next();

        for row in 0..self.nrows as usize {
            let row_start = self.row_idcs[row];
            let row_end = self.row_idcs[row + 1];
            let row_len = row_end - row_start;

            // update row_idcs
            self.row_idcs[row] = new_values.len();

            // move old values
            new_values.extend(old_values_iter.by_ref().take(row_len));
            new_col_idcs.extend(old_col_idcs_iter.by_ref().take(row_len));

            // move value from new column
            if current_col.as_ref().map_or(false, |&(idx, _)| idx as usize == row) {
                let (_, val) = current_col.take().unwrap();
                new_values.push(val);
                new_col_idcs.push(self.ncols);
                current_col = col_iter.next();
            }
        }
        //update after-the-end
        self.row_idcs[self.nrows as usize] = new_values.len();

        self.values = new_values;
        self.col_idcs = new_col_idcs;
        self.ncols += 1;
    }

    /// Return the number of non-zero entries in the given row.
    pub fn row_weight(&self, row : u32) -> u32 {
        (self.row_idcs[(row + 1) as usize] - self.row_idcs[row as usize]) as u32
    }

    /// Extract the last column of the matrix.
    pub fn last_column(self) -> SparseVector<F> {
        let mut values = self.values;
        let mut ret = SparseVector::new(self.nrows, self.field.clone());
        for row in 0..self.nrows as usize {
            let start = self.row_idcs[row];
            let end = self.row_idcs[row + 1];
            if end > start && self.col_idcs[end - 1] + 1 == self.ncols {
                //last entry of this column is non zero
                ret.idcs.push(row as u32);
                ret.values.push(std::mem::replace(&mut values[end - 1], self.field.zero()));
            }
        }
        ret
    }

    /// Extract the reversed last column of the matrix.
    pub fn last_column_rev(self) -> SparseVector<F> {
        let mut values = self.values;
        let mut ret = SparseVector::new(self.nrows, self.field.clone());
        for row in (0..self.nrows as usize).rev() {
            let start = self.row_idcs[row];
            let end = self.row_idcs[row + 1];
            if end > start && self.col_idcs[end - 1] + 1 == self.ncols {
                //last entry of this column is non zero
                ret.idcs.push(self.nrows - (row as u32) - 1);
                ret.values.push(std::mem::replace(&mut values[end - 1], self.field.zero()));
            }
        }
        ret
    }

    /// Format in Mathematica form
    ///
    /// Simply apply `SparseArray@@` to the output in MMA.
    ///
    /// # Example
    /// ```rust
    /// use numerica::tensors::sparse::{SparseMatrix};
    /// use numerica::domains::integer::{IntegerRing, Integer};
    /// let r = IntegerRing::new();
    ///
    /// let mat = SparseMatrix::from_csr(2,3,vec![Integer::new(10),Integer::new(7),Integer::new(13)],vec![0,2,3],vec![0,2,1],r);
    /// assert_eq!(mat.fmt_mma(), "{{{1,1}->10,{1,3}->7,{2,2}->13},{2,3}}");
    ///  ```
    pub fn fmt_mma(&self) -> String {
        let vals = self.row_idcs.windows(2).enumerate() //iterate over rows (with index)
            .flat_map(|(idx, pair)| { //iterate over row entries
                (pair[0]..pair[1]).map(move |i| {
                    //format each element as {idx,col}->val
                    let val = RingPrinter::new(&self.field, &self.values[i]);
                    format!("{{{},{}}}->{}", idx + 1, self.col_idcs[i] + 1, val)
                })
            })
            .join(",");
        //format as {vals, {nrows, ncols}}
        format!("{{{{{}}},{{{},{}}}}}", vals, self.nrows, self.ncols)
    }

    /// Erase zeroes from the values vector.
    fn erase_zeroes(&mut self) -> () {
        let mut pos : usize = 0;
        //iterate over rows
        for row in 0..(self.nrows as usize) {
            let start = self.row_idcs[row];
            let end = self.row_idcs[row + 1];

            //record new start of row
            self.row_idcs[row] = pos;

            for i in start..end {
                if !self.field.is_zero(&self.values[i]) {
                    //move it to correct position
                    self.values[pos] = self.values[i].clone();
                    self.col_idcs[pos] = self.col_idcs[i];
                    pos += 1;
                }
            }
        }
        //write after-the-end pos for last row
        self.row_idcs[self.nrows as usize] = pos;

        //shrink the vector to their actual values
        self.values.truncate(pos);
        self.col_idcs.truncate(pos);
    }
}

impl<F: Field> SparseMatrix<F> {
    /// Solve the linear system `A * x = b`, where `A` is `self` using the Gplu algorithm.
    pub fn solve(mut self, b : SparseVector<F>) -> Result<SparseVector<F>, SparseMatrixError<F>> {
        if self.nrows() != b.len() {
            return Err(SparseMatrixError::ShapeMismatch);
        }
        if self.field != b.field {
            return Err(SparseMatrixError::FieldMismatch);
        }

        let nvars = self.ncols;
        // append b as the last column
		self.append_col(b);

        //perform gplu
        let gplu = Gplu::from_matrix_checked(&self, GpluLMode::None);

        if gplu.is_none() {
            return Err(SparseMatrixError::Inconsistent)
        }

        let mut gplu = gplu.unwrap();

        //check for underdeterminedness
        // rank < nvars
		if gplu.u.nrows() < nvars {
            return Err(SparseMatrixError::Underdetermined {
                rank : gplu.u.nrows() as usize,
                row_reduced_augmented_matrix : gplu.u
            });
        }

        //go to actual rref form
        gplu.back_substitution();

        //solution is the reversed last column of U
        Ok(gplu.u.last_column_rev())
    }
}

impl<F: Field + Sync> SparseMatrix<F>
where F::Element: Sync + Send {
    /// Solve the linear system `A * x = b`, where `A` is `self` using the Gplu algorithm.
    /// The back substitution uses parallelized code.
    pub fn solve_parallel(mut self, b : SparseVector<F>) -> Result<SparseVector<F>, SparseMatrixError<F>> {
        if self.nrows() != b.len() {
            return Err(SparseMatrixError::ShapeMismatch);
        }
        if self.field != b.field {
            return Err(SparseMatrixError::FieldMismatch);
        }

        let nvars = self.ncols;
        // append b as the last column
		self.append_col(b);

        //perform gplu
        let gplu = Gplu::from_matrix_checked(&self, GpluLMode::None);

        if gplu.is_none() {
            return Err(SparseMatrixError::Inconsistent)
        }

        let mut gplu = gplu.unwrap();

        //check for underdeterminedness
        // rank < nvars
		if gplu.u.nrows() < nvars {
            return Err(SparseMatrixError::Underdetermined {
                rank : gplu.u.nrows() as usize,
                row_reduced_augmented_matrix : gplu.u
            });
        }

        //go to actual rref form
        gplu.back_substitution_parallel();

        //solution is the reversed last column of U
        Ok(gplu.u.last_column_rev())
    }
}


impl SparseMatrix<FractionField<IntegerRing>> {
    /// Generate a random SparseMatrix with the given dimensions and number of entries
    pub fn random(nrows : u32, ncols : u32, nentries : usize) -> Self {
        assert!((nrows as usize) * (ncols as usize) > nentries);
        //idea: generate random entry triplets and use the from_triplets constructor
        let mut rng = rand::rng();

        //generate nentries unique coordinates
        let mut pairs : HashSet<(u32,u32)> = HashSet::with_capacity(nentries);
        while pairs.len() < nentries {
            pairs.insert((rng.random_range(0..(nrows-1)), rng.random_range(0..(ncols-1))));
        }

        let f = FractionField::new(IntegerRing::new());

        let mut triplets : Vec<(u32,u32,Fraction<IntegerRing>)> = pairs.into_iter().enumerate().map(
            |(_, (a, b))| (a, b, f.to_element(Integer::new(rng.random::<i64>()), Integer::new(rng.random::<i64>()), true))
        ).collect();

        triplets.sort();
        
        SparseMatrix::from_triplets(nrows, ncols, triplets, f)
    }
}

impl<F: Ring> Neg for SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Negate each entry of the matrix
    fn neg(mut self) -> Self::Output {
        for val in &mut self.values {
            *val = self.field.neg(&*val);
        }

        self
    }
}

impl<F: Ring> Add<&SparseMatrix<F>> for &SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Add two sparse matrices
    fn add(self, rhs: &SparseMatrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.nrows != rhs.nrows {
            panic!(
                "Cannot add sparse matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }
        if self.field != rhs.field {
            panic!(
                "Cannot add sparse matrices over different fields: {} vs {}",
                self.field, rhs.field
            );
        }

        if self.values.is_empty() {
            return rhs.clone();
        }
        if rhs.values.is_empty() {
            return self.clone();
        }

        let mut values = Vec::new();
        let mut col_idcs = Vec::new();
        let mut row_idcs = Vec::with_capacity((self.nrows + 1) as usize);
        row_idcs.push(0);

        //iterate through both matrices simultaneously
        for row in 0..self.nrows as usize {
            let mut lhs_idx = self.row_idcs[row];
            let mut rhs_idx = rhs.row_idcs[row];
            let lhs_end = self.row_idcs[row + 1];
            let rhs_end = rhs.row_idcs[row + 1];

            //iterate through row entries
            while lhs_idx < lhs_end || rhs_idx < rhs_end {
                match (lhs_idx < lhs_end, rhs_idx < rhs_end) {
                    (true, true) => {
                        let lhs_col = self.col_idcs[lhs_idx];
                        let rhs_col = rhs.col_idcs[rhs_idx];

                        if lhs_col == rhs_col {
                            let sum = self.field.add(&self.values[lhs_idx], &rhs.values[rhs_idx]);

                            if !self.field.is_zero(&sum) {
                                col_idcs.push(lhs_col);
                                values.push(sum)
                            }
                            lhs_idx += 1;
                            rhs_idx += 1;
                        } else if lhs_col < rhs_col {
                            col_idcs.push(lhs_col);
                            values.push(self.values[lhs_idx].clone());
                            lhs_idx += 1;
                        } else {
                            col_idcs.push(rhs_col);
                            values.push(rhs.values[rhs_idx].clone());
                            rhs_idx += 1;
                        }
                    },
                    (true, false) => {
                        col_idcs.push(self.col_idcs[lhs_idx]);
                        values.push(self.values[lhs_idx].clone());
                        lhs_idx += 1;
                    },
                    (false, true) => {
                        col_idcs.push(rhs.col_idcs[rhs_idx]);
                        values.push(rhs.values[rhs_idx].clone());
                        rhs_idx += 1;
                    },
                    (false, false) => unreachable!(),
                }
            }

            row_idcs.push(values.len());
        }

        SparseMatrix {
            values,
            row_idcs,
            col_idcs,
            nrows : self.nrows,
            ncols : self.ncols,
            field : self.field.clone(),
        }
    }
}

impl<F: Ring> AddAssign<&SparseMatrix<F>> for SparseMatrix<F> {
    /// Add two sparse matrices in place
    fn add_assign(&mut self, rhs: &SparseMatrix<F>) {
        *self = &*self + rhs;
    }
}

impl<F: Ring> Sub<&SparseMatrix<F>> for &SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Add two sparse matrices
    fn sub(self, rhs: &SparseMatrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.nrows != rhs.nrows {
            panic!(
                "Cannot subtract sparse matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }
        if self.field != rhs.field {
            panic!(
                "Cannot subtract sparse matrices over different fields: {} vs {}",
                self.field, rhs.field
            );
        }

        if self.values.is_empty() {
            return rhs.clone().neg();
        }

        if rhs.values.is_empty() {
            return self.clone();
        }

        let mut values = Vec::new();
        let mut col_idcs = Vec::new();
        let mut row_idcs = Vec::with_capacity((self.nrows + 1) as usize);
        row_idcs.push(0);

        //iterate through both matrices simultaneously
        for row in 0..self.nrows as usize {
            let mut lhs_idx = self.row_idcs[row];
            let mut rhs_idx = rhs.row_idcs[row];
            let lhs_end = self.row_idcs[row + 1];
            let rhs_end = rhs.row_idcs[row + 1];

            //iterate through row entries
            while lhs_idx < lhs_end || rhs_idx < rhs_end {
                match (lhs_idx < lhs_end, rhs_idx < rhs_end) {
                    (true, true) => {
                        let lhs_col = self.col_idcs[lhs_idx];
                        let rhs_col = rhs.col_idcs[rhs_idx];

                        if lhs_col == rhs_col {
                            let sum = self.field.sub(&self.values[lhs_idx], &rhs.values[rhs_idx]);

                            if !self.field.is_zero(&sum) {
                                col_idcs.push(lhs_col);
                                values.push(sum)
                            }
                            lhs_idx += 1;
                            rhs_idx += 1;
                        } else if lhs_col < rhs_col {
                            col_idcs.push(lhs_col);
                            values.push(self.values[lhs_idx].clone());
                            lhs_idx += 1;
                        } else {
                            col_idcs.push(rhs_col);
                            values.push(self.field.neg(&rhs.values[rhs_idx]));
                            rhs_idx += 1;
                        }
                    },
                    (true, false) => {
                        col_idcs.push(self.col_idcs[lhs_idx]);
                        values.push(self.values[lhs_idx].clone());
                        lhs_idx += 1;
                    },
                    (false, true) => {
                        col_idcs.push(rhs.col_idcs[rhs_idx]);
                        values.push(self.field.neg(&rhs.values[rhs_idx]));
                        rhs_idx += 1;
                    },
                    (false, false) => unreachable!(),
                }
            }

            row_idcs.push(values.len());
        }

        SparseMatrix {
            values,
            row_idcs,
            col_idcs,
            nrows : self.nrows,
            ncols : self.ncols,
            field : self.field.clone(),
        }
    }
}

impl<F: Ring> SubAssign<&SparseMatrix<F>> for SparseMatrix<F> {
    /// Subtract two sparse matrices in place
    fn sub_assign(&mut self, rhs: &SparseMatrix<F>) {
        *self = &*self - rhs;
    }
}

impl<F: Ring> Mul<&SparseMatrix<F>> for &SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Multiply two sparse matrices.
    fn mul(self, rhs: &SparseMatrix<F>) -> Self::Output {
        if self.ncols != rhs.nrows {
            panic!(
                "Cannot multiply sparse matrices of non-matching shapes: ({},{}) * ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }
        if self.field != rhs.field {
            panic!(
                "Cannot multiply sparse matrices over different fields: {} vs {}",
                self.field, rhs.field
            );
        }

        let mut values = Vec::new();
        let mut col_idcs = Vec::new();
        let mut row_idcs = Vec::with_capacity((self.nrows + 1) as usize);
        row_idcs.push(0);

        // temporary dense vector to accumulate the output for the row we are working on
        let mut row_acc = vec![self.field.zero(); rhs.ncols as usize];

        //handle row by row in self/LHS
        for row in 0..self.nrows as usize {
            //reset row_acc
            for val in &mut row_acc {
                *val = self.field.zero();
            }

            //iterate over the row elements
            for lhs_idx in self.row_idcs[row]..self.row_idcs[row + 1] {
                let lhs_col = self.col_idcs[lhs_idx];
                let lhs_val = &self.values[lhs_idx];

                //iterate over the corresponding row in RHS
                for rhs_idx in rhs.row_idcs[lhs_col as usize]..rhs.row_idcs[(lhs_col + 1) as usize] {
                    let rhs_col = rhs.col_idcs[rhs_idx];
                    let rhs_val = &rhs.values[rhs_idx];

                    row_acc[rhs_col as usize] = self.field.add(&row_acc[rhs_col as usize], &self.field.mul(lhs_val, rhs_val));
                }
            }

            // push non-zero entries into a new row of the return matrix
            for (col, val) in row_acc.iter().enumerate() {
                if !self.field.is_zero(&val) {
                    col_idcs.push(col as u32);
                    values.push(val.clone());
                }
            }

             row_idcs.push(values.len());
        }

        SparseMatrix {
     	    values,
        	col_idcs,
	        row_idcs,
    	    nrows : self.nrows,
        	ncols : rhs.ncols,
	        field : self.field.clone(),
	    }
    }
}

impl<F: Ring> MulAssign<&SparseMatrix<F>> for SparseMatrix<F> {
    /// Multiply two sparse matrices in place
    fn mul_assign(&mut self, rhs: &SparseMatrix<F>) {
        *self = &*self * rhs;
    }
}

/// An option for `Gplu` of how to handle the L matrix
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GpluLMode {
    /// Construct the full L matrix in the process
    Full,
    /// Construct only the pattern of the L matrix (i.e. don't record the values in the CSR format)
    Pattern,
    /// Don't construct L at all
    None
}

/// Performs an (GP)LU decomposition of a sparse matrix.
///
/// I.e. for a given matrix A we compute L*U = A, where U is upper triangular up to row permutations and L is lower triangular.
/// One may submit a whole system and run the reduction, but one may also choose to submit row by row which is then immediately processed
/// according to the GPLU algorithm, see [https://www-almasty.lip6.fr/~bouillaguet/static/publis/CASC16.pdf].
/// The algorithm is adapted from the SpaSM library, it is essentially a rewrite of the function spasm_LU in commit 965089a of SpaSM.
/// After the LU decomposition, a backsubstitution can be applied to U in order to obtain a RREF form of A (up to row permutations).
///
/// # Type parameters
/// * `F` - the field of the matrix entries
#[derive(Debug, Clone)]
pub struct Gplu<F: Field> {
    /// The U output matrix
    pub(crate) u : SparseMatrix<F>,
    
    /// The L output matrix
    l : SparseMatrix<F>,
    
    /// The pivot positions of U for each column.
    /// I.e. there is a pivot on column j and row pivots[j]. No pivot present if None.
    pivots : Vec<Option<u32>>,
    
    /// Whether to keep the L matrix, just record the pattern or don't record anything at all
    mode : GpluLMode,

    /// How many rows we have already added to the system (and reduced)
    nrows : u32,

    /// Internal variable for spasm's GPLU algorithm used to check for early abort
    defficiency : u32,

    /// Internal variable for spasm's GPLU algorithm, stores the result of the forward solve step
    /// It has length mat.ncols
    x : Vec<F::Element>,

    /// Internal variable for spasm's GPLU algorithm, stores the pattern of the forward solution.
    /// It has length mat.ncols
    xj : Vec<u32>,

    /// Internal variables for spasm's GPLU algorithm, used to count the neighbors already traversed in reach()/dfs().
    /// In spasm, this is part of the xj vector.
    /// It has length mat.ncols
    pstack : Vec<u32>,

    /// Internal variables for spasm's GPLU algorithm, indicates which columns have been seen already in reach()/dfs()
    /// In spasm, this is part of the xj vector.
    /// It has length mat.ncols
    marks : Vec<bool>,
}

impl<F: Field> Gplu<F> {
    /// Construct a new row-by-row Gplu decomposer.
    ///
    /// A new row can be added with `add_row()`.
    pub fn new(ncols : u32, field: F, mode: GpluLMode) -> Gplu<F> {
        Gplu {
            u: SparseMatrix::new(0, ncols, field.clone()),
            x : vec![field.zero(); ncols as usize],
            l: SparseMatrix::new(0, 0, field),
            pivots : vec![None; ncols as usize],
            mode : mode,
            nrows : 0,
            defficiency : 0,
            xj : vec![0; ncols as usize],
            pstack : vec![0; ncols as usize],
            marks : vec![false; ncols as usize],
        }
    }

    /// Construct a new Gplu decomposer that immediately decomposes the given matrix.
    ///
    /// More rows can still be added with `add_row()`.
    pub fn from_matrix(mat : &SparseMatrix<F>, mode: GpluLMode) -> Gplu<F> {
        let mut ret = Gplu {
            u: SparseMatrix::new(0, mat.ncols(), mat.field().clone()),
            x : vec![mat.field().zero(); mat.ncols() as usize],
            l: SparseMatrix::new(0, 0, mat.field().clone()),
            pivots : vec![None; mat.ncols() as usize],
            mode : mode,
            nrows : 0,
            defficiency : 0,
            xj : vec![0; mat.ncols() as usize],
            pstack : vec![0; mat.ncols() as usize],
            marks : vec![false; mat.ncols() as usize],
        };

        for pair in mat.row_idcs.windows(2) {
            ret.gplu_row(&mat.values[pair[0]..pair[1]], &mat.col_idcs[pair[0]..pair[1]]);
        }

        ret
    }

    /// Construct a new Gplu decomposer that immediately decomposes the given matrix and checks if for consistency.
    ///
    /// Checking for consistency means that we return None whenever a new row in `U` is all zero except the last entry.
    /// The idea is that we decompose the matrix `(A|b)` for solving the system `A * x = b`, which becomes unsolvable in this case.
    pub fn from_matrix_checked(mat : &SparseMatrix<F>, mode: GpluLMode) -> Option<Gplu<F>> {
        let mut ret = Gplu {
            u: SparseMatrix::new(0, mat.ncols(), mat.field().clone()),
            x : vec![mat.field().zero(); mat.ncols() as usize],
            l: SparseMatrix::new(0, 0, mat.field().clone()),
            pivots : vec![None; mat.ncols() as usize],
            mode : mode,
            nrows : 0,
            defficiency : 0,
            xj : vec![0; mat.ncols() as usize],
            pstack : vec![0; mat.ncols() as usize],
            marks : vec![false; mat.ncols() as usize],
        };

        for pair in mat.row_idcs.windows(2) {
            if let Some(_) = ret.gplu_row(&mat.values[pair[0]..pair[1]], &mat.col_idcs[pair[0]..pair[1]]) {
                //check last, just added, row for inconsistency
                let start = ret.u.row_idcs[(ret.u.nrows - 1) as usize];
                let end = ret.u.row_idcs[ret.u.nrows as usize];
                if end - start == 1 && ret.u.col_idcs[start as usize] + 1 == ret.u.ncols && ret.u.field.is_zero(&ret.u.values[start as usize]) {
                    //row has only one entry and it's on the last column
                    return None;
                }
            }
        }

        Some(ret)
    }

    /// Return the U matrix
    pub fn u(&self) -> &SparseMatrix<F> {
        &self.u
    }

    /// Return the L matrix
    pub fn l(&self) -> &SparseMatrix<F> {
        &self.l
    }

    /// Return the pivot positions of U for each column.
    /// I.e. there is a pivot on column j and row pivots()[j]. No pivot present if None.
    pub fn pivots(&self) -> &Vec<Option<u32>> {
        &self.pivots
    }

    /// Adds a new row to the system and processes it in the next GPLU step
    ///
    /// A GPLU step is essentially the forward solving of the whole system added until now.
    /// # Parameters
    /// * `values` - the values of the non-zero entries of the row
    /// * `col_idcs` - the column indices of the non-zero entries of the row
    ///
    /// # Return
    /// If the new row is linearly independent of the rest of the system it returns the pivot column index
    /// of the new row after the GPLU step, otherwise None.
    pub fn add_row(&mut self, values : &[F::Element], col_idcs : &[u32]) -> Option<u32> {
        assert_eq!(values.len(), col_idcs.len());

        //run next gplu step
        self.gplu_row(values, col_idcs)
    }

    /// Adds empty columns to the U matrix and updates the pivots accordingly.
    ///
    /// * `col_pos` - Ordered(!) positions where the new columns should be inserted. Each entry must NOT account for previously inserted columns.
    pub fn add_cols(&mut self, col_pos : &Vec<u32>) -> () {
        //update U
        self.u.add_cols(col_pos);
        //update pivots
        let mut new_pivots = Vec::with_capacity(self.pivots.len() + col_pos.len());
        let mut pivots_idx : usize = 0;
        let mut col_pos_idx : usize = 0;

        while pivots_idx < self.pivots.len() || col_pos_idx < col_pos.len() {
            if col_pos_idx < col_pos.len() && pivots_idx == col_pos[col_pos_idx] as usize {
                new_pivots.push(None);
                col_pos_idx += 1;
            } else {
                assert!(pivots_idx < self.pivots.len());
                new_pivots.push(self.pivots[pivots_idx]);
                pivots_idx += 1;
            }
        }
        self.pivots = new_pivots;

        //update x
        self.x.resize(self.u.ncols() as usize, self.u.field().zero());

        //update xj, pstack, marks (make them all zero)
        self.xj.resize(self.u.ncols() as usize, 0);
        self.xj.fill(0);
        self.pstack.resize(self.u.ncols() as usize, 0);
        self.pstack.fill(0);
        self.marks.resize(self.u.ncols() as usize, false);
        self.marks.fill(false);
    }

    /// Applies backsubstitution to the U matrix to bring it into reversed RREF form (i.e. U will be in lower right triangular form).
    ///
    /// We do not keep track of the L matrix, so GpluLMode will be set to `None` and the L matrix will be emptied.
    pub fn back_substitution(&mut self) -> () {
		//idea: make a new Gplu and just add rows in reverse order of their pivot.
        //thats's equivalent to usual Gaussian elimination backsubstitution

        let mut gplu = Gplu::new(self.u.ncols, self.u.field.clone(), GpluLMode::None);

        for col in (0..self.pivots.len()).rev() {
            let row = self.pivots[col];
            if row.is_none() {
                continue;
            }
            let row = row.unwrap();
            let start = self.u.row_idcs[row as usize];
            let end = self.u.row_idcs[(row + 1) as usize];

            gplu.add_row(&self.u.values[start..end], &self.u.col_idcs[start..end]);
        }

        *self = gplu;
    }
    
    /// Apply the GPLU algorithm to the given row
    ///
    /// # Return
    /// The pivot column of the new row in U if it was a linearly independent row.
    fn gplu_row(&mut self, values : &[F::Element], col_idcs: &[u32]) -> Option<u32> {
        if self.nrows - self.defficiency == std::cmp::min(self.u.ncols(), self.nrows + 1) {
            //full rank reached
            self.nrows += 1;
            return None;
        } //else

        //check whether the row can be taken directly into U
        let mut directly_pivotal = !values.is_empty(); //empty row check
        if directly_pivotal {
            //check whether one of the entries in the new row is on a pivot column
            for col_idx in col_idcs {
                if let Some(_) = self.pivots[*col_idx as usize] {
                    directly_pivotal = false;
                    break;
                }
            }
        }
        if directly_pivotal {
            //yes, we can directly take it into U!
            //record pivot
            let pivot_col = col_idcs[0];
            self.pivots[pivot_col as usize] = Some(self.nrows - self.defficiency);

            //copy the whole row into U (and divide by leading coefficient)
            let leading_coeff = &values[0];
            let leading_coeff_inv = self.u.field.inv(&leading_coeff);
            self.u.nrows += 1;
            self.u.row_idcs.push(self.u.row_idcs.last().unwrap() + values.len());
            self.u.col_idcs.extend_from_slice(&col_idcs);
            if self.u.field.is_one(&leading_coeff_inv) {
	            self.u.values.extend_from_slice(&values);
            } else {
                self.u.values.extend(values.iter().map(|val| self.u.field.mul(val, &leading_coeff_inv)));
            }

            //also compute L if wanted
            match self.mode {
                GpluLMode::Full => {
                    //put a 1 on the diagonal
                    self.l.col_idcs.push(self.nrows - self.defficiency);
                    self.l.values.push(leading_coeff.clone());
                    //finish the row
                    self.l.row_idcs.push(self.l.col_idcs.len());
                    //update nrows/ncols
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                },
                GpluLMode::Pattern => {
                    //put an entry on the diagonal
                    self.l.col_idcs.push(self.nrows - self.defficiency);
                    //finish the row
                    self.l.row_idcs.push(self.l.col_idcs.len());
                    //update nrows/ncols
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                },
                GpluLMode::None => ()//nothing to be done
            }
            self.nrows += 1;
            return Some(pivot_col);
        }

        //triangular solve: x * U = mat[row]
        let top = Gplu::sparse_forward_solve(values, col_idcs, &self.u, &self.pivots, &mut self.x, &mut self.xj, &mut self.pstack, &mut self.marks);

        //find pivot and dispatch coeffs into U
        let mut pivot : Option<u32> = None;
        for px in top..self.u.ncols() {
            //x[j] is generically nonzero
            let j = self.xj[px as usize];

            //if x[j] == 0 (accidental cancellation) we just ignore it
            if self.u.field.is_zero(&self.x[j as usize]) {
                continue;
            }
            if self.pivots[j as usize].is_none() {
                //column is not yet pivotal
                //better than current pivot
                if pivot.map_or(true, |jj| j < jj) {
                    pivot = Some(j);
                }
            } else {
                //self.l.row_idcs will be updated later
                match self.mode {
                    GpluLMode::Full => {
                        // x[j] is the entry L[i, pivots[j] ]
                        self.l.col_idcs.push(self.pivots[j as usize].unwrap());
                        self.l.values.push(self.x[j as usize].clone());
                    },
                    GpluLMode::Pattern => {
                        self.l.col_idcs.push(self.pivots[j as usize].unwrap());
                    },
                    GpluLMode::None => () //nothing to be done
                }
            }
        }

        //pivot found?
        if pivot.is_some() {
            let pivot = pivot.unwrap();
            // L[i, i] <-- x[pivot], last entry of the row
            match self.mode {
                GpluLMode::Full => {
                    self.l.col_idcs.push(self.nrows - self.defficiency);
                    self.l.values.push(self.x[pivot as usize].clone());
                    //finish the row
                    self.l.row_idcs.push(self.l.col_idcs.len());
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                },
                GpluLMode::Pattern => {
                    self.l.col_idcs.push(self.nrows - self.defficiency);
                    //finish the row
                    self.l.row_idcs.push(self.l.col_idcs.len());
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                },
                GpluLMode::None => () //notheng to be done
            }
            //record new pivot
            self.pivots[pivot as usize] = Some(self.nrows - self.defficiency);

            //fill in U
            //we first collect in vector to sort them before actually inserting
            let mut new_row : Vec<(u32, F::Element)> = Vec::new();

            //pivot must be the first entry in U[i]
            new_row.push((pivot, self.u.field.one()));

            //send the remaining non-pivot coefficients into new row
            let beta = self.u.field.inv(&self.x[pivot as usize]);
            for px in top..self.u.ncols() {
                let j = self.xj[px as usize];

                if self.pivots[j as usize].is_none() {
                    let val = self.u.field.mul(&self.x[j as usize], &beta);
                    if !self.u.field.is_zero(&val) {
                        new_row.push((j,val));
                    }
                }
            }

            //sort
            new_row.sort_unstable_by_key(|(col, _)| *col);
            //move into actual U
            self.u.values.reserve(new_row.len());
            self.u.col_idcs.reserve(new_row.len());
            for (col, val) in new_row {
                self.u.col_idcs.push(col);
                self.u.values.push(val);
            }
            
			//finish the new row in U
            self.u.row_idcs.push(self.u.values.len());
            self.u.nrows += 1;

            self.nrows += 1;
            return Some(pivot);
        }
        //else: need to finish L
        match self.mode {
            GpluLMode::Full => {
                //finish the row
                self.l.row_idcs.push(self.l.col_idcs.len());
                self.l.nrows += 1;
            },
            GpluLMode::Pattern => {
                //finish the row
                self.l.row_idcs.push(self.l.col_idcs.len());
                self.l.nrows += 1;
            },
            GpluLMode::None => () //nothing to be done
        }

        //row is linearly dependent, nothing to do, but record defficiency
        self.defficiency += 1;
        self.nrows += 1;

        None
	}

    /// Perform a forward solution of the eqn x * U = sparse_row
    ///
    /// Helper function of gplu_row().
    /// We take even member variables as argument as we sometimes (e.g. in the parallel back substitution) want to use other vectors.
    /// The solution is scattered in the dense vector x, and its pattern is given in xj[top..ncols],
    /// where top is the return value.
    /// The precise semantics is as follows. Define
    ///    x_a = { j in [0..ncols] : pivots[j] == None }
    ///    x_b = { j in [0..ncols] : pivots[j] == Some(_) }
    /// Then x_b * U + x_a == sparse_row. It follows that x * U == y has a solution iff x_a is empty.
    /// This requires that the pivots in U are all equal to 1.
    /// (Technically it does not require that the pivot are the first entry of the row, but we always have that...)
    ///
    /// # Arguments
    /// * `values` - The non-zero values of the sparse row on the RHS of the equation.
    /// * `col_idcs` - The column indices of the non-zero values of the sparse row on the RHS of the equation.
    /// * `u`, `pivots`, `x`, `xj`, `pstack`, `marks` - Objects that are or correspond to the member variables of Gplu with the same names
    fn sparse_forward_solve(values : &[F::Element], col_idcs : &[u32], u : &SparseMatrix<F>, pivots : &Vec<Option<u32>>, x : &mut Vec<F::Element>,
                            xj : &mut Vec<u32>, pstack : &mut Vec<u32>, marks : &mut Vec<bool>) -> u32 {
        //compute non-zero pattern of x
        let top = Gplu::reach(&col_idcs, &u, &pivots, xj, pstack, marks);

        //clear x and copy sparse_row into x, i.e. x = sparse_row
        for px in top..u.ncols {
            x[xj[px as usize] as usize] = u.field.zero();
        }
        for (col_idx, val) in col_idcs.iter().zip(values.iter()) {
            x[*col_idx as usize] = val.clone();
        }

        //iterate over the precomputed pattern of x
        for px in top..u.ncols() {
            let j = xj[px as usize];//x[j] is generically nonzero (except for accidental numerical cancellations)
            //locate corresponding pivot if there is any
            let i = pivots[j as usize];

            if i.is_none() {
                continue;
            }

            let i = i.unwrap();
            //the pivot on row i is 1, so we just have to multiply by -x[j]
            let backup = x[j as usize].clone();

            let start = u.row_idcs[i as usize];
            let end = u.row_idcs[(i + 1) as usize];
            let beta = u.field.neg(&x[j as usize]);
            Gplu::scatter(x, &beta, &u.values[start..end], &u.col_idcs[start..end], &u.field);
            debug_assert!(u.field.is_zero(&x[j as usize]));
            x[j as usize] = backup;
        }

        top
    }

    /// Compute the reachability of columns of U from all column indices in the sparse_row.
    ///
    /// Helper function of sparse_forward_solve().
    ///
    /// The variables xj, pstack, and marks of self must be of size u.ncols and zeroed out the first time this function is called.
    /// On output, the set of reachable columns is written in xj[top..u.ncols], where top is the return value.
    /// xj, pstack, and marks remain in a usable state for further calls of this function ond doesn't need to be zeroed out.
    /// # Arguments
    /// * `col_idcs` - The column indices of the non-zero values of the sparse row on the RHS of the equation.
    /// * `u`, `pivots`, `xj`, `pstack`, `marks` - Objects that are or correspond to the member variables of Gplu with the same names
    fn reach(col_idcs : &[u32], u : &SparseMatrix<F>, pivots : &Vec<Option<u32>>, xj : &mut Vec<u32>, pstack : &mut Vec<u32>, marks : &mut Vec<bool>) -> u32 {
        let mut top = u.ncols();

	    //iterate over the kth row of mat. For each column index j present in mat[k] check if j is in the pattern
    	//(i.e. if it's marked). If not, start a DFS from j and add to the pattern all columns reachable from j
        for j in col_idcs {
	        if !marks[*j as usize] {
    	        top = Gplu::dfs(*j, top, &u, &pivots, xj, pstack, marks);
        	}
	    }
    	//unmark all marked nodes
        for px in top..u.ncols() {
            marks[xj[px as usize] as usize] = false;
	    }
        top
    }

    /// Depth-first-search along alternating paths of a bipartite graph representation of U.
    ///
    /// If a column j is pivotal (pivots[j] != None), then move to the row (call it i)
    /// containing the pivot; explore columns adjacent to row i, depth first.
    /// The traversal starts at col_start.
    ///
    /// At the end, the list of traversed nodes is in xj[top..u.ncols], where top is the return value.
    /// # Arguments
    /// * `u`, `pivots`, `xj`, `pstack`, `marks` - Objects that are or correspond to the member variables of Gplu with the same names
    fn dfs(jstart : u32, mut top : u32, u : &SparseMatrix<F>, pivots : &Vec<Option<u32>>, xj : &mut Vec<u32>, pstack : &mut Vec<u32>, marks : &mut Vec<bool>) -> u32 {
        //initialize the recursion stack (columns waiting to be traversed)
        //he stack is held at the beginning of xj, and has 'head' elements
        let mut head : u32 = 0;
        xj[head as usize] = jstart;

        loop {
            //get j from the top of the recursion stack
            let j = xj[head as usize];
            let i = pivots[j as usize];

            if !marks[j as usize] {
                //mark column j as seen and initialize pstack. This is done only once
                marks[j as usize] = true;
                pstack[head as usize] = 0;
            }

            if i.is_none() {
                //push initial column in the output stack and pop out from the recursion stack
                top -= 1;
                xj[top as usize] = xj[head as usize];
                if head == 0 {
                    break;
                }//else
                head -= 1;
                continue;
            }

            let i = i.unwrap();

            //size of row i
            let row_weight_i = u.row_weight(i);

            //examine all yet-unseen entries of row i
            let mut k : u32 = pstack[head as usize];
            while k < row_weight_i {
                let px = u.row_idcs[i as usize] + (k as usize);
                let j2 = u.col_idcs[px];
                if marks[j2 as usize] {
                    //step
                    k += 1;
                    continue;
                }
                //interrupt the enumeration of entries of row i and start DFS from column j2
                pstack[head as usize] = k + 1;
                head += 1;
                xj[head as usize] = j2;
                break;
            }
            if k == row_weight_i {
                //row i fully examined; push initial column in the output stack and pop it from the recursion stack
                top -= 1;
                xj[top as usize] = xj[head as usize];
                if head == 0 {
                    break;
                }
                head -= 1;
            }
            
        }
        top
    }

    /// Compute x = x + beta * sparse_row, writing it directly into x.
    ///
    /// Helper function for the GPLU algorithm
    /// # Arguments
    /// * `x` - the vector x into which we scatter.
    /// * `beta` - the scalar which we multiply into the sparse row
    /// * `values` - he non-zero values of the sparse row on the RHS of the equation.
    /// * `col_idcs` - he column indices of the non-zero values of the sparse row on the RHS of the equation.
    /// * `field` - the field to be used for the arithmetics.
    fn scatter(x : &mut Vec<F::Element>, beta : &F::Element, values : &[F::Element], col_idcs: &[u32], field : &F) {
        for (val, col) in values.iter().zip(col_idcs.iter()) {
            x[*col as usize] = field.add(&field.mul(beta, &val), &x[*col as usize]);
        }
    }
}


impl<F: Field + Sync> Gplu<F>
where F::Element: Sync + Send {
    /// Applies backsubstitution to the U matrix to bring it into reversed RREF form (i.e. U will be in lower right triangular form).
    ///
    /// We do not keep track of the L matrix, so GpluLMode will be set to `None` and the L matrix will be emptied.
    /// This version employs a parallel algorithm, which though in total does more work than the serial version.
    pub fn back_substitution_parallel(&mut self) -> () {
        //erase L if necessary
        match self.mode {
            GpluLMode::Full | GpluLMode::Pattern => {
                self.l.values.clear();
                self.l.col_idcs.clear();
                self.l.row_idcs.clear();
                self.l.row_idcs.push(0);
                self.l.nrows = 0;
                self.l.ncols = 0;
                self.mode = GpluLMode::None;
            },
            GpluLMode::None => () //nothing to be done
        }
        
        //we need some local objects for each thread: pivots, x, xj, pstack, marks
        let new_rows : Vec<_> = self.pivots.par_iter().enumerate().rev().filter_map(|(col_idx, opt)| opt.as_ref().map(|val| (col_idx, val))).map_init(
            || {(
                //create the local versions of pivots, x, xj, pstack, marks
                self.pivots.clone(),
                vec![self.u.field.one(); self.u.ncols as usize],
                vec![0; self.u.ncols as usize],
                vec![0; self.u.ncols as usize],
                vec![false; self.u.ncols as usize],
            )},
            |(pivots, x, xj, pstack, marks), (col_idx, row) | {
                //remove the pivot on the row we are acting on from the local pivots
                pivots[col_idx] = None;
                
                let start = self.u.row_idcs[*row as usize];
                let end = self.u.row_idcs[(row + 1) as usize];
                let top = Gplu::sparse_forward_solve(&self.u.values[start..end], &self.u.col_idcs[start..end], &self.u, pivots, x, xj, pstack, marks);

                //reset the pivot in the local pivots
                pivots[col_idx] = Some(*row);

                //collet col_idcs and values of new row in a single vector
                let mut new_row : Vec<(u32, F::Element)> = Vec::new();

                //put the pivot first
                new_row.push((col_idx as u32, self.u.field.one()));

                //send the remaining non-pivots into the new row
                let beta = self.u.field.inv(&x[col_idx]);
                for px in top..self.u.ncols() {
                    let j = xj[px as usize];
                    if pivots[j as usize].is_none() {
                        let val = self.u.field.mul(&x[j as usize], &beta);
                        if !self.u.field.is_zero(&val) {
                            new_row.push((j,val));
                        }
                    }
                }
                
                //sort
                new_row.sort_unstable_by_key(|(col, _)| *col);
                //return
                new_row
                
            }
        ).collect();

        debug_assert_eq!(new_rows.len(),self.u.nrows as usize);

        //move into u
        self.u.values.clear();
        self.u.col_idcs.clear();
        self.u.row_idcs.clear();

        //count elements
        let total_vals : usize = new_rows.iter().map(|row| row.len()).sum();

        self.u.values.reserve(total_vals);
        self.u.col_idcs.reserve(total_vals);
        self.u.row_idcs.reserve(new_rows.len() + 1);
        
        self.u.row_idcs.push(0);
        for row in new_rows.into_iter() {
            for (col_idx, val) in row.into_iter() {
                self.u.values.push(val);
                self.u.col_idcs.push(col_idx);
            }
            self.u.row_idcs.push(self.u.values.len());
        }
        
    }
}

#[cfg(test)]
mod tests {
    use crate::domains::{
        integer::{IntegerRing, Integer},
        rational::{FractionField, Fraction}
    };

    use crate::tensors::sparse::{SparseMatrix, SparseVector, Gplu, GpluLMode};

    #[test]
    fn random_gplu_backsubs() {
        let mat = SparseMatrix::<FractionField<IntegerRing>>::random(100, 100, 100);

        let mut gplu = Gplu::from_matrix(&mat, GpluLMode::Full);

        //check L.U == A (also checking multiplication and subtraction)
        assert_eq!(&(gplu.l() * gplu.u()), &mat);
        assert_eq!(&(gplu.l() * gplu.u()) - &mat, SparseMatrix::new(mat.nrows(), mat.ncols(), FractionField::new(IntegerRing::new())));

        let mut gplu2 = gplu.clone();

        //check the two versions of back_substitution against each other
        gplu.back_substitution();
        gplu2.back_substitution_parallel();
        assert_eq!(gplu.u(), gplu2.u());
    }

    #[test]
    fn row_by_row_gplu() {
        let r = IntegerRing::new();
        let f = FractionField::new(r);

        let mut gplu = Gplu::new(6, f.clone(), GpluLMode::Full);
        let mut mat = SparseMatrix::new(0, 6, f.clone());

        {
	        let mut values : Vec<Fraction<IntegerRing>> = Vec::new();
    	    let mut col_idcs : Vec<u32> = Vec::new();

        	values.push(f.to_element(Integer::new(3),Integer::new(1),false));
            col_idcs.push(1);
            values.push(f.to_element(Integer::new(7),Integer::new(1),false));
            col_idcs.push(2);
            values.push(f.to_element(Integer::new(13),Integer::new(1),false));
            col_idcs.push(5);

            gplu.add_row(&values, &col_idcs);
            mat.add_row(values, col_idcs);
        }

        {
	        let mut values : Vec<Fraction<IntegerRing>> = Vec::new();
    	    let mut col_idcs : Vec<u32> = Vec::new();

        	values.push(f.to_element(Integer::new(-2),Integer::new(1),false));
            col_idcs.push(0);
            values.push(f.to_element(Integer::new(14),Integer::new(1),false));
            col_idcs.push(3);
            values.push(f.to_element(Integer::new(-27),Integer::new(1),false));
            col_idcs.push(4);

            gplu.add_row(&values, &col_idcs);
            mat.add_row(values, col_idcs);
        }

        {
	        let mut values : Vec<Fraction<IntegerRing>> = Vec::new();
    	    let mut col_idcs : Vec<u32> = Vec::new();

        	values.push(f.to_element(Integer::new(23),Integer::new(1),false));
            col_idcs.push(1);
            values.push(f.to_element(Integer::new(18),Integer::new(1),false));
            col_idcs.push(2);
            values.push(f.to_element(Integer::new(6),Integer::new(1),false));
            col_idcs.push(4);

            gplu.add_row(&values, &col_idcs);
            mat.add_row(values, col_idcs);
        }

        //check L.U == A (also checking multiplication and subtraction)
        assert_eq!(&(gplu.l() * gplu.u()), &mat);
        assert_eq!(&(gplu.l() * gplu.u()) - &mat, SparseMatrix::new(mat.nrows(), mat.ncols(), f.clone()));
        //check U
        assert_eq!(gplu.u().fmt_mma(), "{{{1,2}->1,{1,3}->7/3,{1,6}->13/3,{2,1}->1,{2,4}->-7,{2,5}->27/2,{3,3}->1,{3,5}->-18/107,{3,6}->299/107},{3,6}}");

        gplu.back_substitution();
        //check rref
        assert_eq!(gplu.u().fmt_mma(), "{{{1,3}->1,{1,5}->-18/107,{1,6}->299/107,{2,2}->1,{2,5}->42/107,{2,6}->-234/107,{3,1}->1,{3,4}->-7,{3,5}->27/2},{3,6}}");
    }

    #[test]
    fn all_at_once_gplu() {
        let r = IntegerRing::new();
        let f = FractionField::new(r);

        let mut mat = SparseMatrix::new(0, 6, f.clone());

        {
	        let mut values : Vec<Fraction<IntegerRing>> = Vec::new();
    	    let mut col_idcs : Vec<u32> = Vec::new();

        	values.push(f.to_element(Integer::new(3),Integer::new(1),false));
            col_idcs.push(1);
            values.push(f.to_element(Integer::new(7),Integer::new(1),false));
            col_idcs.push(2);
            values.push(f.to_element(Integer::new(13),Integer::new(1),false));
            col_idcs.push(5);

            mat.add_row(values, col_idcs);
        }

        {
	        let mut values : Vec<Fraction<IntegerRing>> = Vec::new();
    	    let mut col_idcs : Vec<u32> = Vec::new();

        	values.push(f.to_element(Integer::new(-2),Integer::new(1),false));
            col_idcs.push(0);
            values.push(f.to_element(Integer::new(14),Integer::new(1),false));
            col_idcs.push(3);
            values.push(f.to_element(Integer::new(-27),Integer::new(1),false));
            col_idcs.push(4);

            mat.add_row(values, col_idcs);
        }

        {
	        let mut values : Vec<Fraction<IntegerRing>> = Vec::new();
    	    let mut col_idcs : Vec<u32> = Vec::new();

        	values.push(f.to_element(Integer::new(23),Integer::new(1),false));
            col_idcs.push(1);
            values.push(f.to_element(Integer::new(18),Integer::new(1),false));
            col_idcs.push(2);
            values.push(f.to_element(Integer::new(6),Integer::new(1),false));
            col_idcs.push(4);

            mat.add_row(values, col_idcs);
        }

        let mut gplu = Gplu::from_matrix(&mat, GpluLMode::Full);

        //check L.U == A (also checking multiplication and subtraction)
        assert_eq!(&(gplu.l() * gplu.u()), &mat);
        assert_eq!(&(gplu.l() * gplu.u()) - &mat, SparseMatrix::new(mat.nrows(), mat.ncols(), f.clone()));
        //check U
        assert_eq!(gplu.u().fmt_mma(), "{{{1,2}->1,{1,3}->7/3,{1,6}->13/3,{2,1}->1,{2,4}->-7,{2,5}->27/2,{3,3}->1,{3,5}->-18/107,{3,6}->299/107},{3,6}}");
        
        //check rref
        gplu.back_substitution();
        assert_eq!(gplu.u().fmt_mma(), "{{{1,3}->1,{1,5}->-18/107,{1,6}->299/107,{2,2}->1,{2,5}->42/107,{2,6}->-234/107,{3,1}->1,{3,4}->-7,{3,5}->27/2},{3,6}}");
    }

    #[test]
    fn solve() {
        let f = FractionField::new(IntegerRing::new());
        // sparse 5x5 matrix triplets
        let triplets = vec![
            // row, col, entry
            (0, 0, f.to_element(Integer::new(1), Integer::new(1), false)),
            (0, 2, f.to_element(Integer::new(2), Integer::new(1), false)),
            (1, 1, f.to_element(Integer::new(1), Integer::new(1), false)),
            (1, 3, f.to_element(Integer::new(3), Integer::new(1), false)),
            (2, 2, f.to_element(Integer::new(1), Integer::new(1), false)),
            (2, 4, f.to_element(Integer::new(4), Integer::new(1), false)),
            (3, 3, f.to_element(Integer::new(1), Integer::new(1), false)),
            (4, 0, f.to_element(Integer::new(2), Integer::new(1), false)),
            (4, 4, f.to_element(Integer::new(1), Integer::new(1), false)),
        ];

        //sparse vector pairs
        let pairs = vec![
            (0, f.to_element(Integer::new(3), Integer::new(1), false)),
            (1, f.to_element(Integer::new(5), Integer::new(1), false)),
            (2, f.to_element(Integer::new(7), Integer::new(1), false)),
            (3, f.to_element(Integer::new(2), Integer::new(1), false)),
            (4, f.to_element(Integer::new(8), Integer::new(1), false)),
        ];

        let mat = SparseMatrix::from_triplets(5, 5, triplets, f.clone());

        let b = SparseVector::from_pairs(5, pairs, f);

        println!("mat = SparseArray@@{};", mat.fmt_mma());
        println!("b = SparseArray@@{};", b.fmt_mma());

        let res = mat.solve(b);

        match res {
            Ok(value) => assert_eq!(value.fmt_mma(), "{{{1,1}->53/17,{2,1}->-1,{3,1}->-1/17,{4,1}->2,{5,1}->30/17},{5,1}}"),
            Err(_) => assert!(false)
        }
    }
}
