# ndarray-householder-qr

[![Build Status](https://travis-ci.org/scijs/ndarray-householder-qr.svg?branch=1.0.0)](https://travis-ci.org/scijs/ndarray-householder-qr) [![npm version](https://badge.fury.io/js/ndarray-householder-qr.svg)](http://badge.fury.io/js/ndarray-householder-qr)

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
    a = ndarray([0,0])      // unknown model parameters

    v = qr.workVector(m,n),
    A = vander(x,n);

qr.triangularize( A, v );
qr.solve( A, v, y, a );

// result: a = ndarray([ 1, 1 ]) --> y = 1 * x + 1
```

After this calculation, the factorization can be reused to solve for other inputs:

```javascript
var moreData = ndarray([2,3,4]);

qr.solve( A, v, moreData, a );

// result: a = ndarray([ 2, 1 ]) --> y = 1 * x + 2
```


## Usage

##### `triangularize( A, v )`
Computes the in-place triangularization of A (which is then R), returning a sequence of packed Householder reflectors in v. There shouldn't be too many cases where you need to use v directly since the magic of the Householder QR factorization is that you can implicitly calculate both Q' * b and Q * x without ever explicitly construcing Q. v must be a one-dimensional vector with length at least `m*n-n*(n-1)/2`.

##### `multByQ( v, n, x )`
Compute the product Q * x in-place, replacing x with Q * x. v is the result of the Householder triangularization, n is the width of the original matrix.

##### `multByQinv( v, n, b )`
Compute the product Q^-1 * x in-place, replacing b with Q^-1 * b. v is the result of the Householder triangularization.

##### `constructQ( Q, v )`
**Incomplete**
Given a series of Householder reflectors, construct the matrix Q by applying the reflectors to a sequence of unit vectors.

##### `factor( A, Q )`
**Incomplete**
Compute the in-place QR factorization of A, storing R in A and outputting Q in Q.

##### `solve( R, v, b, x )`
Use the previously-calculated triangularization to find the vector x that minimizes the L-2 norm of (Ax - b). Note that the vector b is modified in the process.
- `R` is an upper-triangular matrix calculated in the `triangularize` step. The dimensions must be at least n x n, so it's fine if the in-place factorized m x n matrix A is used.
- `v` is the work vector that stores the householder reflectors.
- `b` is the input vector of length m.
- `x` is the output vector of length n.


#####


## Credits
(c) 2015 Ricky Reusser. MIT License
