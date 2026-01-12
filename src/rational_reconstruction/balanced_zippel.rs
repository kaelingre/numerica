//! Methods to run the "Balanced Zippel" reconstruction algorithm proposed in hep-th/2409.19099

use crate::{
    domains::{
        finite_field::{Zp64, FiniteFieldElement},
        Ring, RingOps, Field,
    }
};


/// A partially reconstructed univariate rational function over Zp64.
///
/// We follow the algorithm in hep-th/1904.00009, eqns. (22)-(25) by computing and storing th b_{i,j} coefficients
/// One must add value by value using add_value(). The user must take care of storing the sample points {t_1,t_2,...}.
#[derive(Debug)]
struct ThieleGuess {
    /// Whether the rational reconstruction has terminated already
    pub done : bool,

    /// Storing the b_i coefficients
    pub bis : Vec<FiniteFieldElement<u64> >,
}

impl ThieleGuess {
    /// Get the b_i coefficient (which already must have been computed by adding the corresponding evalution f(t_{i+1}) to self)
    fn bi(&self, i : usize) -> &FiniteFieldElement<u64> {
        //b_i is at position i * (i + 3) /2 in the collected bijs
        &self.bis[i]
    }

    /// Evaluate the current guess at the point `x`
    ///
    /// # Arguments
    /// * `x` - The point at which to evaluate the current guess.
    /// * `eval_points` - All sample points {t_1,t_2,...} at which the function has been evaluated so far.
    ///                   They are needed since we don't store them internally.
    ///                   It may contain more points than the ones already evaluated.
    /// * `field` - The finite field over which we are working.
    fn eval(&self, x : FiniteFieldElement<u64>, sample_points : &[FiniteFieldElement<u64>], field : &Zp64) -> FiniteFieldElement<u64> {
        if self.bis.is_empty() {
            return field.zero();
        }
        //compute from inside to outside
        let mut ret = *self.bis.last().unwrap();
        for (bi, ti) in self.bis[..self.bis.len()-1].iter().rev().zip(sample_points[..self.bis.len()-1].iter().rev()) {
            ret = field.add(bi, &field.div(&field.sub(x, *ti), &ret));
        }

        ret
    }

    /// Adds a next evaluation value that triggers the computation of the next b_i.
    ///
    /// # Arguments
    /// * `value` - The new evaluation value.
    /// * `eval_points` - All sample points {t_1,t_2,...} at which the function has been evaluated so far.
    ///                   It may contain more points than the ones already evaluated.
    /// * `field` - The finite field over which we are working.
    fn add_value(&mut self, value : FiniteFieldElement<u64>, sample_points : &[FiniteFieldElement<u64>], field : &Zp64) -> bool {
        debug_assert!(sample_points.len() > self.bis.len());
        if self.done {
            return true;
        }
        //check whether we are done by comparing the new value to the evaluation of the current guess at the newest point
        if self.eval(sample_points[self.bis.len()], sample_points, field) == value {
            // we're done, guess evaluates to the same as the newly given value
            self.done = true;
            return true;
        }

        //compute the next b_i: compute recursively b_{i,0}, b_{i,1}, b_{i,2}, ..., b_{i,i} = b_i
        let i = self.bis.len();
        let mut bij = value;
        for j in 0..i {
            bij = field.div(&field.sub(sample_points[i], sample_points[j]), &field.sub(&bij, &self.bis[j]));
        }
        self.bis.push(bij);
        
        false
    }
}
