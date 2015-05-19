'use strict'

var blas = require('ndarray-blas-level1'),
    fill = require('ndarray-fill')

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
    s = blas.nrm2(Ajmj)
    ajj = Ajmj.get(0)
    dj = ajj > 0 ? -s : s
    d.set(j,dj)
    s = Math.sqrt( s * (s + Math.abs(ajj)) )
    if( s === 0 ) return false
    Ajmj.set(0,ajj - dj)
    blas.scal( 1/s, Ajmj)

    for(i=j+1; i<n; i++) {
      Ajmi = A.pick(null,i).lo(j)
      s = - blas.dot( Ajmj, Ajmi )
      blas.axpy( s, Ajmj, Ajmi )
    }
  }

  return true
}


var multiplyByQ = function multiplyByQ ( A, b ) {
  var j, Ajmj, yk, s, n = A.shape[1]

  for(j=n-1; j>=0; j--) {
    Ajmj = A.pick(null,j).lo(j)
    yk = b.lo(j)
    s = -blas.dot(Ajmj,yk)
    blas.axpy( s, Ajmj, yk)
  }
  return true
}

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

var constructQ = function( QR, Q ) {
  var j, Qj, n=Q.shape[1]

  fill(Q,function(i,j){ return i===j ? 1 : 0})

  for(j=0; j<n; j++) {
    Qj = Q.pick(null,j)
    multiplyByQ( QR, Qj )
  }

  return true
}

//var factorize = function( A, Q, R ) {
//}


var solve = function( QR, d, x ) {
  var m,n, dot = blas.dot

  if( QR.dimension !== 2 ) {
    throw new TypeError('triangularize():: dimension of input matrix must be 2.')
  }

  m = QR.shape[0]
  n = QR.shape[1]

  multiplyByQinv( QR, x )

  // Copied from TRSV, except replacing the diagonal with d:
  x.set( n-1, x.get(n-1)/d.get(n-1,n-1) )
  for(var i=n-2; i>=0; i--) {
    x.set(i, (x.get(i) - dot(QR.pick(i,null).lo(i+1), x.lo(i+1))) / d.get(i) )
  }

  return true
}


exports.triangularize = triangularize
exports.multiplyByQ = multiplyByQ
exports.multiplyByQinv = multiplyByQinv
exports.constructQ = constructQ
//exports.factorize = factorize
exports.solve = solve
