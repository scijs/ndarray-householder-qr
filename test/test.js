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

  var m,n,A,v,b,y,A0,x;

  beforeEach(function() {
    m=3;
    n=2;
    // I sat down and worked this example by hand, so I'm pretty confident in
    // the numbers as long as they follow the algorithm....

    x = ndarray([0,1,2]);

    A0 = vander(x,2);
    A = vander(x,2);
    v = householder.workVector(m,n);
    b = ndarray([1,2,3]);
    y = pool.zeros([n])
  });

  describe("triangularize",function() {
    it('calculates the correct Householder reflectors',function() {
      var vExpected = ndarray([0.888, 0.325, 0.325, 0.793, 0.609])

      householder.triangularize(A,v);
      assert( ndtest.approximatelyEqual( vExpected, v, 1e-3 ) );
    });

    it('R is upper-triangular',function() {
      householder.triangularize(A,v);
      assert( ndtest.matrixIsUpperTriangular(A,1e-8) );
    });

    it('calculates the correct R matrix',function() {
      var RExpected = ndarray([ -1.732, -1.732, 0, -1.414, 0, 0],[3,2])

      householder.triangularize(A,v);
      assert( ndtest.approximatelyEqual( RExpected, A, 1e-3 ) );
    });
  });

  describe("multiplyByQinv()", function() {
    it('succeeds',function() {
      householder.triangularize(A,v);
      assert( householder.multiplyByQinv(v,b,y) );
    });

    it('calculates the correct product',function() {
      var QinvbExpected = ndarray([-3.464, -1.414, 0])

      householder.triangularize(A,v);
      householder.multiplyByQinv(v, n, b);

      assert( ndtest.approximatelyEqual( QinvbExpected, b, 1e-3 ) );
    });

  });

  describe("solve()", function() {
    it('succeeds',function() {
      assert( householder.triangularize(A,v) );
      assert( householder.solve(A,v,b,y) );
    });

    it('calculates the right answer',function() {
      var xExpected = ndarray([1,1]);

      householder.triangularize(A,v);
      householder.solve(A,v,b,y);

      assert( ndtest.approximatelyEqual( xExpected, y, 1e-4 ) );
    });

    it('calculates the right answer when reused',function() {
      var xExpected = ndarray([1,1]);

      householder.triangularize(A,v);

      householder.solve(A,v, ndarray([1,2,3]), y);
      householder.solve(A,v, ndarray([1,2,3]), y);

      assert( ndtest.approximatelyEqual( xExpected, y, 1e-4 ) );
    });

  });

  xdescribe("multiplyByQ()", function() {

    it('succeeds',function() {
      householder.triangularize(A,v);
      assert( householder.multiplyByQ(v,y,b) );
    });

  });


});
