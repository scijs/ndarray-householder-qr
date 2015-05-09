'use strict';

var householder = require('../index.js'),
    assert = require('chai').assert,
    ndarray = require('ndarray'),
    gemm = require("ndgemm"),
    blas2 = require('ndarray-blas-level2'),
    ops = require('ndarray-ops'),
    show = require('ndarray-show'),
    ndtest = require('ndarray-tests'),
    fill = require('ndarray-fill'),
    pool = require('ndarray-scratch'),
    vander = require('ndarray-vandermonde');


describe("Householder QR", function() {

  var m,n,A,v,b,x,A0;

  beforeEach(function() {
    m=3;
    n=2;
    // I sat down and worked this example by hand, so I'm pretty confident in
    // the numbers as long as they follow the algorithm....
    A0 = ndarray(new Float64Array([1,0,1,1,1,2]), [m,n])
    A = ndarray(new Float64Array([1,0,1,1,1,2]), [m,n])
    v = householder.workVector(m,n);
    b = ndarray([1,2,3]);
    x = pool.zeros([n])
  });

  describe("triangularize",function() {
    it('calculates the correct Householder reflectors',function() {
      householder.triangularize(A,v);
      var vExpected = ndarray([0.888, 0.325, 0.325, 0.793, 0.609])
      assert( ndtest.approximatelyEqual( vExpected, v, 1e-3 ) );
    });

    it('R is upper-triangular',function() {
      householder.triangularize(A,v);
      assert( ndtest.matrixIsUpperTriangular(A,1e-8) );
    });

    it('calculates the correct R matrix',function() {
      householder.triangularize(A,v);
      var RExpected = ndarray([ -1.732, -1.732, 0, -1.414, 0, 0],[3,2])
      assert( ndtest.approximatelyEqual( RExpected, A, 1e-3 ) );
    });

  });


  xdescribe("multiplyByQinv()", function() {

    it('succeeds',function() {
      householder.triangularize(A,v);
      assert( householder.multiplyByQinv(v,b,x) );
    });

  });

  xdescribe("multiplyByQ()", function() {

    it('succeeds',function() {
      householder.triangularize(A,v);
      assert( householder.multiplyByQ(v,x,b) );
    });

  });

  xdescribe("solve()", function() {

    it('succeeds',function() {
      householder.solve(A,b,x);

      blas2.gemv(1, A0, x, 0, b);
      console.log('reconstruct=',show(b));
    });

  });

});
