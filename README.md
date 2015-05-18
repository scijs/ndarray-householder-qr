# ndarray-householder-qr

[![Build Status](https://travis-ci.org/scijs/ndarray-householder-qr.svg?branch=master)](https://travis-ci.org/scijs/ndarray-householder-qr) [![npm version](https://badge.fury.io/js/ndarray-householder-qr.svg)](http://badge.fury.io/js/ndarray-householder-qr) [![Dependency Status](https://david-dm.org/scijs/ndarray-householder-qr.svg)](https://david-dm.org/scijs/ndarray-householder-qr)

A module for calculating the in-place [QR decomposition](http://en.wikipedia.org/wiki/QR_decomposition) of an ndarray using Householder triangularization

## Introduction

The algorithm is the Householder QR Factorization algorithm as found on p. 73 of Trefethen and Bau's [Numerical Linear Algebra](http://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617). In pseudocode, the algorithm is:

```
for k = 1 to n
  x = A[k:m,k]
  v_k = sign(x_1) ||x||_2 e_1 + x
  v_k = v_k / ||v_k||_2
  A[k:m,k:n] = A[k:m,k:n] - 2 v_k (v_k^* A[k:m,k:n])
```

## Example

A straightforward example of the usefulness of QR factorization is the solution of least squares problems. To fit the model `y = a0 * x + a1` to the data points `[x1,y1] = [0,1]`, `[x2,y2] = [1,2]`, `[x3,y3] = [2,3]`: 

```javascript
var qr = require('ndarray-householder-qr'),
    vander = require('ndarray-vandermonde'),
    
    m = 3,
    n = 2,

    x = ndarray([0,1,2]),   // independent variable
    y = ndarray([1,2,3]),   // data points

    d = pool.zeros([n]),
    A = vander(x,n);

qr.triangularize( A, d );
qr.solve( A, d, y );

// result: y = ndarray([ 1, 1, 0 ]) --> y = 1 * x + 1
```

After this calculation, the factorization can be reused to solve for other inputs:

```javascript
var y2 = ndarray([2,3,4]);

qr.solve( A, v, y2 );

// result: y = ndarray([ 2, 1, 0 ]) --> y = 1 * x + 2
```


## Usage

##### `triangularize( A, d )`
Computes the in-place triangularization of `A`, returning the Householder reflectors in the lower-triangular portion of `A` (including the diagonal) and `R` in the upper-triangular portion of `A` (excluding diagonal) with the diagonal of `R` stored in `d`. `d` must be a one-dimensional vector with length at least `n`.

##### `multByQ( A, x )`
**Incomplete**
Compute the product Q * x in-place, replacing x with Q * x. v is the result of the Householder triangularization, n is the width of the original matrix.

##### `multByQinv( A, x )`
`A` is the in-place triangularized matrix. Compute the product `Q^-1 * x` in-place, replacing x with `Q^-1 * x`. Since the product is shorter than `x` for m > n, the entries of `x` from n+1 to m will be zero.

##### `constructQ( A )`
**Incomplete**
Given a series of Householder reflectors, construct the matrix Q by applying the reflectors to a sequence of unit vectors.

##### `factor( A, Q )`
**Incomplete**
Compute the in-place QR factorization of A, storing R in A and outputting Q in Q.

##### `solve( A, d, x )`
Use the previously-calculated triangularization to find the vector x that minimizes the L-2 norm of (Ax - b). Note that the vector b is modified in the process.
- `A` is the in-place triangularized matrix computed by `triangularize`
- `d` is the diagonal of `R` computed by `triangularize`
- `x` is the input vector of length m. The answer is computed in-place in the first n entries of `x`. The remaining entries are zero.


## Credits
(c) 2015 Ricky Reusser. MIT License
