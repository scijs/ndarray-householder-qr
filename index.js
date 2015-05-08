'use strict';

var blas = require('ndarray-blas-level1');

var householderTriangularize = function householderTriangularize( A, v ) {
  var i,k,n,m;

  if( A.dimension !== 2 ) {
    throw new TypeError('householderTriangularize():: dimension of input matrix must be 2.');
  }

  m = A.shape[0];
  n = A.shape[1];

  if( m < n ) {
    throw new TypeError('householderTriangularize():: In input matrix A, number of rows m must be greater than number of column n.');
  }

  if( v === undefined || (v.shape!==undefined && v.shape[0] < m*n - n*(n-1)/2) ) {
    throw new TypeError('householderTriangularize():: v must be a vector of length >= m*n-n*(n-1)/2 given A is a matrix of size m x n.');
  }

  var vlo, x0, sgn, x, nrm, Akmkn, alpha;

  for( vlo=0, k=0; k<n; k++ ) {
    // Get this section of the packed output array:
    x = v.lo(vlo).hi(m-k);

    // v = A[k:m,k]
    blas.copy( A.pick(null,k).lo(k), x );

    // vk = sign(x1) ||x||_2 e1 + x
    nrm = blas.nrm2(x);
    x0 = x.get(0);
    sgn = x0 < 0 ? -1 : 1;
    x.set(0, x0 + sgn*nrm);

    // vk = vk / ||vk||_2
    nrm = blas.nrm2(x);
    blas.scal( 1/nrm, x);

    // A[k:m,k:n] = A[k:n,k:n] - 2 * vk * (vk^* * A[k:m,k:n])
    
    // Compute alpha = -2 * vk^* * A[k:m,k:n]:
    Akmkn = A.lo(k,k);
    alpha = - 2 * blas.dot(x, Akmkn.pick(null,0));

    for(i=0; i<n-k; i++) {
      blas.axpy(alpha, x, Akmkn.pick(null,i));
    }

    // Increment the current position in the packed output vector v:
    vlo += m-k;
  }


  return true;
};

module.exports = householderTriangularize;
