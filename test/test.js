'use strict';

var qr = require('../index.js'),
    assert = require('chai').assert,
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    gemm = require("ndgemm"),
    ops = require('ndarray-ops'),
    show = require('ndarray-show'),
    ndtest = require('ndarray-tests'),
    fill = require('ndarray-fill');


describe("householderTriangularize()", function() {

  var m,n,A,v;

  beforeEach(function() {
    m=7;
    n=4;
    A = pool.zeros([m,n],'float64');
    v = pool.zeros([m*n-n*(n-1)/2],'float64');

    fill(A,function(i,j) { return Math.sqrt(2+i+j); });

    qr(A,v);
  });

  it('returns a triangularized matrix in place of A',function() {
    console.log(show(A.transpose(1,0)));
    console.log(show(v));
    assert( ndtest.upperTriangular(A,1e-8) );
  });

});
