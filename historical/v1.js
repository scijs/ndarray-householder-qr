'use strict';

var blas = require('ndarray-blas-level1');
var trsv = require('ndarray-blas-trsv');
var pool = require('ndarray-scratch');

// var show = require('ndarray-show');

var triangularize = function triangularize (A, v) {
  var i, k, n, m;

  if (A.dimension !== 2) {
    throw new TypeError('triangularize():: dimension of input matrix must be 2.');
  }

  m = A.shape[0];
  n = A.shape[1];

  if (m < n) {
    throw new TypeError('triangularize():: In input matrix A, number of rows m must be greater than number of column n.');
  }

  if (v === undefined || (v.shape !== undefined && v.shape[0] < m * n - n * (n - 1) / 2)) {
    throw new TypeError('triangularize():: v must be a vector of length >= m*n-n*(n-1)/2 given A is a matrix of size m x n.');
  }

  var vlo, x0, sgn, x, nrm, Akmkn, alpha, Akmi;

  for (vlo = 0, k = 0; k < n; k++) {
    // Get this section of the packed output array:
    x = v.lo(vlo).hi(m - k);

    // v = A[k:m,k]
    blas.copy(A.pick(null, k).lo(k), x);

    // vk = sign(x1) ||x||_2 e1 + x
    nrm = blas.nrm2(x);
    x0 = x.get(0);
    sgn = x0 < 0 ? -1 : 1;
    x.set(0, x0 + sgn * nrm);

    // vk = vk / ||vk||_2
    nrm = blas.nrm2(x);
    if (nrm === 0) return false;
    blas.scal(1 / nrm, x);

    // A[k:m,k:n] = A[k:n,k:n] - 2 * vk * (vk^* * A[k:m,k:n])

    // Compute alpha = -2 * vk^* * A[k:m,k:n]:
    Akmkn = A.lo(k, k);

    for (i = 0; i < n - k; i++) {
      Akmi = Akmkn.pick(null, i);
      alpha = -2 * blas.dot(x, Akmi);
      blas.axpy(alpha, x, Akmi);
    }

    // Increment the current position in the packed output vector v:
    vlo += m - k;
  }

  return true;
};

var multiplyByQ = function multiplyByQ (v, n, x) {
  var m, k, xkm, vk, alpha;

  m = x.shape[0];

  for (k = n - 1; k >= 0; k--) {
    xkm = x.lo(k);

    // Get this section of the packed output array:
    vk = x.lo(k).hi(m - k);

    // calculate -2 * vk' * x[k:m]
    alpha = -2 * blas.dot(vk, xkm);

    // Update b[k:m] = b[k:m] - 2 * v * (vk' * b[k:m])
    blas.axpy(alpha, vk, xkm);
  }
  return true;
};

var multiplyByQinv = function multiplyByQinv (v, n, b) {
  var k, vlo, m, bkm, vk, alpha;

  m = b.shape[0];

  for (vlo = 0, k = 0; k < n; k++) {
    bkm = b.lo(k);

    // Get this section of the packed output array:
    vk = v.lo(vlo).hi(m - k);

    // calculate -2 * vk' * b[k:m]
    alpha = -2 * blas.dot(vk, bkm);

    // Update b[k:m] = b[k:m] - 2 * v * (vk' * b[k:m])
    blas.axpy(alpha, vk, bkm);

    // Move to the next section of the packed v array
    vlo += m - k;
  }

  return true;
};

// var constructQ = function( Q, v ) {
// };

// var factor = function( ) {
// };

var workVectorLength = function (m, n) {
  return m * n - n * (n - 1) / 2;
};

var workVector = function (m, n, dtype) {
  console.warn('workVector is deprecated. Please use workVectorLength() and allocate a one-dimensional ndarray of that size instead.');
  if (dtype === undefined) {
    dtype = 'float64';
  }
  return pool.zeros([m * n - n * (n - 1) / 2], dtype);
};

var solve = function (R, v, b, x) {
  var n;

  if (R.dimension !== 2) {
    throw new TypeError('triangularize():: dimension of input matrix must be 2.');
  }

  n = R.shape[1];

  multiplyByQinv(v, n, b);

  blas.copy(b.hi(n), x);
  trsv(R.lo(0, 0).hi(n, n), x);

  return true;
};

exports.triangularize = triangularize;
exports.multiplyByQ = multiplyByQ;
exports.multiplyByQinv = multiplyByQinv;
exports.workVectorLength = workVectorLength;
exports.workVector = workVector;
// exports.constructQ = constructQ;
// exports.factor = factor;
exports.solve = solve;
