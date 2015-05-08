# ndarray-householder-qr

[![Build Status](https://travis-ci.org/scijs/ndarray-householder-qr.svg?branch=1.0.0)](https://travis-ci.org/scijs/ndarray-householder-qr) [![npm version](https://badge.fury.io/js/ndarray-householder-qr.svg)](http://badge.fury.io/js/ndarray-householder-qr)

A module for calculating the in-place [QR decomposition of a matrix](http://en.wikipedia.org/wiki/QR_decomposition)

## Introduction

The algorithm is the Householder QR Factorization algorithm as found on p. 58 of Trefethen and Bau's [Numerical Linear Algebra](http://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617). In pseudocode, the algorithm is:

```
for k = 1 to n
  x = A[k:m,k]
  v_k = sign(x_1) ||x||_2 e_1 + x
  v_k = v_k / ||v_k||_2
  A[k:m,k:n] = A[k:m,k:n] - 2 v_k (v_k^* A[k:m,k:n])

## Usage

The algorithm currently only calculates the in-place QR decomposition and returns true on successful completion.

```
var qr = require('ndarray-householder-qr'),
    pool = require('ndarray-scratch');

var A = ndarray( new Float64Array([1,2,7,4,5,1,7,4,9]), [3,3] );
var R = pool.zeros( A.shape, A.dtype );

qr( A, R );
```

Then the product A * R is approximately equal to the original matrix.

## Credits
(c) 2015 Ricky Reusser. MIT License
