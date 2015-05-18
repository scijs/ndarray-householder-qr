'use strict'

var blas = require('ndarray-blas-level1'),
    fill = require('ndarray-fill')
    //ndshow = require('ndarray-show')

/*function show(label,data) {
  if( typeof data === 'number' ) {
    console.log(label+' = '+data)
  } else if( data.dimension === 1 ) {
    console.log(label+' = '+ndshow(data))
  } else {
    console.log(label+' =\n'+ndshow(data.transpose(1,0))+'\n')
  }
}*/

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

  Ajmj = A.pick(null,0)
  Ajmi = A.pick(null,0)
  var AjmiOff = Ajmi.offset
  var AjmjInc = A.stride[0] + A.stride[1]
  var AjmiInc1 = A.stride[0]
  var AjmiInc2 = A.stride[1]
  var dInc = d.stride[0]
  var dOff = d.offset

  for( j=0; j<n; j++, Ajmj.shape[0]--, Ajmj.offset+=AjmjInc, AjmiOff+=AjmiInc1, dOff+=dInc) {
    s = blas.nrm2(Ajmj)
    ajj = Ajmj.data[Ajmj.offset]
    dj = ajj > 0 ? -s : s
    d.data[dOff] = dj
    s = Math.sqrt( s * (s + Math.abs(ajj)) )
    if( s === 0 ) return false
    Ajmj.data[Ajmj.offset] = ajj - dj
    blas.scal( 1/s, Ajmj)

    for(i=j+1, Ajmi.offset=AjmiOff + AjmiInc2*(j+1); i<n; i++, Ajmi.offset+=AjmiInc2) {
      s = - blas.dot( Ajmj, Ajmi )
      blas.axpy( s, Ajmj, Ajmi )
    }
  }

  return true
}


var multiplyByQ = function multiplyByQ ( A, b ) {
  var j, Ajmj, yk, n=A.shape[1], m=A.shape[0]

  Ajmj = A.pick(null,n-1).lo(n-1)
  yk = b.lo(n-1)

  var AjmjOffInc = A.stride[0] + A.stride[1]
  var ykInc = yk.stride[0]

  for(j=n-1; j>=0; j--, Ajmj.offset-=AjmjOffInc, Ajmj.shape[0]++, yk.offset-=ykInc, yk.shape[0]++ ) {
    blas.axpy( -blas.dot(Ajmj,yk), Ajmj, yk)
  }

  return true
}
//console.log('i='+j,'shape=',Ajmj.shape,'stride=',Ajmj.stride,'offset=',Ajmj.offset)

var multiplyByQinv = function multiplyByQinv( A, b ) {
  var j, Ajmj, yk, n=A.shape[1], m=A.shape[0]

  Ajmj = A.pick(null,0).lo(0)
  yk = b.lo(0)

  var AjmjOffInc = A.stride[0] + A.stride[1]
  var ykInc = yk.stride[0]

  for(j=0; j<n; j++, Ajmj.offset+=AjmjOffInc, Ajmj.shape[0]--, yk.offset+=ykInc, yk.shape[0]--) {
    blas.axpy( -blas.dot(Ajmj,yk), Ajmj, yk)
  }

  return true
}

var constructQ = function( QR, Q ) {
  var j, Qj, n=Q.shape[1], m=Q.shape[0]

  fill(Q,function(i,j){ return i===j ? 1 : 0})

  for(j=0; j<n; j++ ) {
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
