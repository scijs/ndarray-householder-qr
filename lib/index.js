'use strict'

var blas = require('ndarray-blas-level1'),
    trsv = require('ndarray-blas-trsv'),
    pool = require('ndarray-scratch'),
    ndshow = require('ndarray-show')

function show(label,data) {
  if( typeof data === 'number' ) {
    console.log(label+' = '+data)
  } else if( data.dimension === 1 ) {
    console.log(label+' = '+ndshow(data))
  } else {
    console.log(label+' =\n'+ndshow(data.transpose(1,0))+'\n')
  }
}
//var show = require('ndarray-show')

var triangularize = function triangularize( A, d ) {
  var j,m,n,i, s, ajj, Ajmj, dj, alpha, Ajmi, fak

  if( A.dimension !== 2 ) {
    throw new TypeError('triangularize():: dimension of input matrix must be 2.')
  }

  m = A.shape[0]
  n = A.shape[1]

  if( m < n ) {
    throw new TypeError('triangularize():: In input matrix A, number of rows m must be greater than number of column n.')
  }

  for( j=0; j<n; j++ ) {
    Ajmj = A.pick(null,j).lo(j)
    s = blas.nrm2(Ajmj);
    ajj = Ajmj.get(0)
    dj = ajj > 0 ? -s : s
    d.set(j,dj)
    s = Math.sqrt( s * (s + Math.abs(ajj)) )
    if( s === 0 ) return false;
    Ajmj.set(0,ajj - dj)
    blas.scal( 1/s, Ajmj)

    for(i=j+1; i<n; i++) {
      Ajmi = A.pick(null,i).lo(j)
      s = - blas.dot( Ajmj, Ajmi )
      blas.axpy( s, Ajmj, Ajmi )
    }
  }

  return true;
};


var multiplyByQ = function multiplyByQ ( v, n, x ) {
  var m, k, vlo, xkm, vk, alpha;

  m = x.shape[0];

  vlo = n*(2*m - n - 1)/2;

  for(k=n-1; k>=0; k--) {
    xkm = x.lo(k);

    // Get this section of the packed output array:
    vk = x.lo(k).hi(m-k);

    // calculate -2 * vk' * x[k:m]
    alpha = - 2 * blas.dot(vk, xkm);

    // Update b[k:m] = b[k:m] - 2 * v * (vk' * b[k:m])
    blas.axpy( alpha, vk, xkm );

    vlo -= m-k+1;
  }
  return true;
};

var multiplyByQinv = function multiplyByQinv( A, b ) {
  var j, Ajmj, yk, s, n = A.shape[1]

  for(j=0; j<n; j++) {
    Ajmj = A.pick(null,j).lo(j)
    yk = b.lo(j)
    s = -blas.dot(Ajmj,yk)
    blas.axpy( s, Ajmj, yk)
  }

  return true
}

//var constructQ = function( Q, v ) {
//};

//var factor = function( ) {
//};


var solve = function( QR, d, x ) {
  var m,n, dot = blas.dot;

  if( QR.dimension !== 2 ) {
    throw new TypeError('triangularize():: dimension of input matrix must be 2.');
  }

  m = QR.shape[0];
  n = QR.shape[1];

  multiplyByQinv( QR, x );

  // Copied from TRSV, except replacing the diagonal with d:
  x.set( n-1, x.get(n-1)/d.get(n-1,n-1) );
  for(var i=n-2; i>=0; i--) {
    x.set(i, (x.get(i) - dot(QR.pick(i,null).lo(i+1), x.lo(i+1))) / d.get(i) );
  }

  return true;
};


exports.triangularize = triangularize;
exports.multiplyByQ = multiplyByQ;
exports.multiplyByQinv = multiplyByQinv;
//exports.constructQ = constructQ;
//exports.factor = factor;
exports.solve = solve;
