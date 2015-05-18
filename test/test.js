'use strict';

var householder = require('../lib'),
    assert = require('chai').assert,
    ndarray = require('ndarray'),
    gemm = require("ndgemm"),
    blas = require('ndarray-blas-level1'),
    ndshow = require('ndarray-show'),
    ndtest = require('ndarray-tests'),
    fill = require('ndarray-fill'),
    pool = require('ndarray-scratch'),
    vander = require('ndarray-vandermonde');

function show(label,data) {
  if( typeof data === 'number' ) {
    console.log(label+' = '+data)
  } else if( data.dimension === 1 ) {
    console.log(label+' = '+ndshow(data))
  } else {
    console.log(label+' =\n'+ndshow(data.transpose(1,0))+'\n')
  }
}

describe("Householder QR", function() {

  var m,n,A,d,b,y,A0,x,Q,Qreduced

  beforeEach(function() {
    m=3
    n=2
    // I sat down and worked this example by hand, so I'm pretty confident in
    // the numbers as long as they follow the algorithm....

    x = ndarray([0,1,2])

    A0 = vander(x,2)
    A = vander(x,2)
    Q = pool.zeros([m,m])
    Qreduced = pool.zeros([m,n])
    d = pool.zeros([n])
    b = ndarray([1,2,3])
    y = pool.zeros([n])
  })

  afterEach(function() {
    pool.free(d)
    pool.free(y)
  })

  describe("triangularize",function() {
    beforeEach(function() {
      householder.triangularize(A,d)
    })

    it('calculates the correct Householder reflectors',function() {
      var v1 = ndarray([1.256, 0.46, 0.46])
      var v2 = ndarray([1.122, 0.861])

      assert( ndtest.approximatelyEqual( v1, A.pick(null,0), 1e-3 ) )
      assert( ndtest.approximatelyEqual( v2, A.pick(null,1).lo(1), 1e-3 ) )
    })

    it('calculates the correct R matrix',function() {
      var dExp = ndarray([-1.732, -1.414])
      assert( ndtest.approximatelyEqual( dExp, d, 1e-3 ) )
      assert( Math.abs( A.get(0,1) - (-1.732) ) < 1e-3 )
    })
  })

  describe("multiplyByQinv()", function() {
    it('succeeds',function() {
      householder.triangularize(A,d)
      assert( householder.multiplyByQinv(A, b) )
    })

    it('calculates the correct product',function() {
      var QinvbExpected = ndarray([-3.464, -1.414, 0])

      householder.triangularize(A,d)
      householder.multiplyByQinv(A, b)

      assert( ndtest.approximatelyEqual( QinvbExpected, b, 1e-3 ) )
    })
  })

  describe("multiplyByQ()", function() {

    it('succeeds',function() {
      householder.triangularize(A,d)
      assert( householder.multiplyByQ(A, b) )
    })

    it('calculates the correct product',function() {
      var x = ndarray([-3.464, -1.414, 0])
      var QbExpected = ndarray([1,2,3])

      householder.triangularize(A,d)
      householder.multiplyByQ(A, x)

      assert( ndtest.approximatelyEqual( QbExpected, x, 1e-3 ) )
    })

  })

  describe("constructQ()", function() {

    beforeEach(function() {
      householder.triangularize(A,d)
    })

    it('calculates Q for the full QR factorization',function(){
      assert( householder.constructQ(A,Q) )
      assert( ndtest.matrixOrthogonal(Q,1e-3) )
    })

    it('calculates Q for the reduced QR factorization',function(){
      assert( householder.constructQ(A,Qreduced) )
      assert( ndtest.matrixColsNormalized(Qreduced,1e-3) )
      assert( ndtest.matrixColsAreOrthogonal(Qreduced,1e-3) )
    })


  })


  describe("solve()", function() {
    it('succeeds',function() {
      assert( householder.triangularize(A,d) )
      assert( householder.solve(A,d,b,y) )
    })

    it('calculates the right answer',function() {
      var xExpected = ndarray([1,1])

      householder.triangularize(A,d)
      householder.solve(A,d,b)

      assert( ndtest.approximatelyEqual( xExpected, b.hi(n), 1e-4 ) )
    })

    it('calculates the right answer when reused',function() {
      var xExpected = ndarray([1,1])

      householder.triangularize(A,d)

      var y1 = ndarray([1,2,3])
      var y2 = ndarray([1,2,3])
      householder.solve(A,d,y1)
      householder.solve(A,d,y2)

      assert( ndtest.approximatelyEqual( xExpected, y1.hi(n), 1e-4 ) )
      assert( ndtest.approximatelyEqual( xExpected, y2.hi(n), 1e-4 ) )
    })

  })


})
