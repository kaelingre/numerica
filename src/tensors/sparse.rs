//! Sparse linear algebra methods using matrices and vectors.
//!
//! # Example
//!
//! ```rust
//! use spired::util::MathematicaFormat;
//! use spired::sparse::{SparseMatrix};
//! use symbolica::domains::integer::{IntegerRing, Integer};
//! let r = IntegerRing::new();
//!
//! let mat = SparseMatrix::from_csr(2,3,vec![Integer::new(10),Integer::new(7),Integer::new(13)],vec![0,2,3],vec![0,2,1],r);
//! assert_eq!(mat.fmt_mma(), "{{{1,1}->10,{1,3}->7,{2,2}->13},{2,3}}");

use itertools::Itertools;

use crate::{
    domains::{
        Ring, Field, RingPrinter
    }
};

/// A sparse matrix in compressed sparse row (CSR) format
///
/// We keep each row sorted at all times. Makes the algorithms slightly faster.
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

    /// Format in Mathematica format
    ///
    /// Simply apply `SparseArray@@` to the output in MMA.
    ///
    /// # Example
    /// ```rust
    /// use spired::util::MathematicaFormat;
    /// use spired::sparse::{SparseMatrix};
    /// use symbolica::domains::integer::{IntegerRing, Integer};
    /// let r = IntegerRing::new();
    ///
    /// let mat = SparseMatrix::from_csr(2,3,vec![Integer::new(10),Integer::new(7),Integer::new(13)],vec![0,2,3],vec![0,2,1],r);
    /// assert_eq!(mat.fmt_mma(), "{{{1,1}->10,{1,3}->7,{2,2}->13},{2,3}}");
    ///  ```
    fn fmt_mma(&self) -> String {
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

/// Performs an LU decomposition on a sparse matrix.
///
/// I.e. for a given matrix A we compute L*U = P*A, where U is upper triangular, L is lower triangular and P is a permutation matrix,
/// meaning U is the RREF of A up to permutations of rows.
/// One may submit a whole system and run the reduction, but one may also choose to submit row by row which is the immediately process
/// according to the GPLU algorithm.
/// The algorithm is adapted from the SpaSM library, it is essentially the function spasm_LU in commit 965089a of SpaSM.
///
/// # Type parameters
/// * `F` - the field of the matrix entries
#[derive(Debug)]
pub struct Gplu<F: Field> {
    /// The original system that is to be decomposed
    mat : SparseMatrix<F>,
    
    /// The U output matrix
    u : SparseMatrix<F>,
    
    /// The L output matrix
    l : SparseMatrix<F>,
    
    /// The pivots positions of U for each column.
    /// I.e. there is a pivot on column j and row pivot[j].
    pivots : Vec<Option<u32> >,
    
    /// The pivot columns of U
    /// The inverse of pivots: The vector has one entry per row in U indicating the position of the pivot in that row
    pivot_cols : Vec<u32>,

    /// Whether to keep the L matrix, just record the pattern or don't record anything at all
    mode : GpluLMode,

    /// Internal variable for spasm's GPLU algorithm used to check for early abort
    defficiency : u32,

    /// Internal variable for spasm's GPLU algorithm
    x : Vec<F::Element>,
    
}

impl<F: Field> Gplu<F> {
    /// Construct a new row-by-row Gplu reducer with the given mode for the L matrix
    ///
    /// A new row can be added with `add_row()`.
    pub fn new(ncols : u32, field: F, mode: GpluLMode) -> Gplu<F> {
        Gplu {
            mat: SparseMatrix::new(0, ncols, field.clone()),
            u: SparseMatrix::new(0, ncols, field.clone()),
            x : vec![field.zero() ; ncols as usize],
            l: match mode {
                GpluLMode::Full | GpluLMode::Pattern => SparseMatrix::new(0, ncols, field),
                GpluLMode::None => SparseMatrix::new(0, 0, field)
            },
            pivots : vec![None; ncols as usize],
            pivot_cols : Vec::new(),
            mode : mode,
            defficiency : 0,
        }
    }

    /// Return the U matrix
    pub fn u(&self) -> &SparseMatrix<F> {
        &self.u
    }

    /// Return the L matrix
    pub fn l(&self) -> &SparseMatrix<F> {
        &self.l
    }

    /// Adds a new row to the system and processes it in the next GPLU step
    ///
    /// # Parameters
    /// * `values` - the values of the non-zero entries of the row
    /// * `col_idcs` - the column indices of the non-zero entries of the row
    ///
    /// # Return
    /// If the new row is linearly independent of the rest of the system it returns the pivot column index
    /// of the new row after the GPLU step, otherwise None.
    pub fn add_row(&mut self, values : Vec<F::Element>, col_idcs : Vec<u32>) -> Option<u32> {
        assert_eq!(values.len(), col_idcs.len());

        //add new row to mat
        self.mat.row_idcs.push(self.mat.row_idcs.last().unwrap() + values.len());
        self.mat.values.extend(values);
        self.mat.col_idcs.extend(col_idcs);
        self.mat.nrows += 1;

        //run next gplu step
        self.gplu_row(self.mat.nrows() - 1)
    }

    /// Adds empty columns to all the matrices and updates the pivots accordingly.
    ///
    /// * `col_pos` - Ordered(!) positions where the new columns should be inserted. Each entry must NOT account for previously inserted columns.
    pub fn add_cols(&mut self, col_pos : &Vec<u32>) -> () {
        //update the three matrices
        self.mat.add_cols(col_pos);
        self.u.add_cols(col_pos);
        match self.mode {
            GpluLMode::Full | GpluLMode::Pattern => self.l.add_cols(col_pos),
            GpluLMode::None => ()
        };
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
        
        //update pivot_cols, row by row
        for row in 0..self.pivot_cols.len() {
            let mut col_pos_it : usize = 0;
            while col_pos_it < col_pos.len() && col_pos[col_pos_it] <= self.pivot_cols[row] {
                col_pos_it += 1;
            }
            self.pivot_cols[row] += col_pos_it as u32;
        }

        //update x
        self.x.resize(self.mat.ncols() as usize, self.mat.field().zero());
    }

    /// Apply the GPLU algorithm to the given row
    ///
    /// # Return
    /// The pivot column of the new row in U if it was a linearly independet row.
    fn gplu_row(&mut self, row : u32) -> Option<u32> {
        if row - self.defficiency == std::cmp::min(self.mat.ncols(), self.mat.nrows()) {
            //full rank reached
            return None;
        } //else

        let row_idcs = &self.mat.row_idcs;
        let col_idcs = &self.mat.col_idcs;
        let values = &self.mat.values;
        let field = &self.mat.field;

        //check whether the row can be taken directly into U
        let mut directly_pivotal = row_idcs[(row + 1) as usize] > row_idcs[row as usize]; //empty row check
        if directly_pivotal {
            //check whether one of the entries in the new row is on a pivot column
            for pos in row_idcs[row as usize]..row_idcs[(row + 1) as usize] {
                if let Some(_) = self.pivots[col_idcs[pos] as usize] {
                    directly_pivotal = false;
                    break;
                }
            }
        }
        if directly_pivotal {
            //yes, we can directly take it into U!
            //record pivot
            let pivot_col = col_idcs[row_idcs[row as usize]];
            self.pivots[pivot_col as usize] = Some(row - self.defficiency);
            self.pivot_cols.push(pivot_col);

            //copy the whole row into U (and divide by leading coefficient)
            let leading_coeff_inv = field.inv(&values[row_idcs[row as usize]]);
            self.u.nrows += 1;
            self.u.row_idcs.push(self.u.row_idcs.last().unwrap());
            let row_idcs_last = self.u.row_idcs.last_mut().unwrap();

            
        }

        

        None
	}
}
