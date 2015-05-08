# ndarray-householder-qr

[![Build Status](https://travis-ci.org/scijs/ndarray-householder-qr.svg?branch=1.0.0)](https://travis-ci.org/scijs/ndarray-householder-qr) [![npm version](https://badge.fury.io/js/ndarray-householder-qr.svg)](http://badge.fury.io/js/ndarray-householder-qr)

A module for calculating the in-place [QR decomposition of a matrix](http://en.wikipedia.org/wiki/QR_decomposition) of ndarrays using Householder triangularization

## Introduction

The algorithm is the Householder QR Factorization algorithm as found on p. 73 of Trefethen and Bau's [Numerical Linear Algebra](http://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617). In pseudocode, the algorithm is:

```
for k = 1 to n
  x = A[k:m,k]
  v_k = sign(x_1) ||x||_2 e_1 + x
  v_k = v_k / ||v_k||_2
  A[k:m,k:n] = A[k:m,k:n] - 2 v_k (v_k^* A[k:m,k:n])
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

##### `solve( A, b )`
**Incomplete**
Compute the in-place QR factorization of A and compute A^-1 b = x, storing the result in the original vector b.

#####


## Credits
(c) 2015 Ricky Reusser. MIT License
