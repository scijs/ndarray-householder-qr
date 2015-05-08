'use strict';

var householder = require('../index.js'),
    assert = require('chai').assert,
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    gemm = require("ndgemm"),
    ops = require('ndarray-ops'),
    show = require('ndarray-show'),
    ndtest = require('ndarray-tests'),
    fill = require('ndarray-fill');


describe("triangularize()", function() {

  var m,n,A,v;

  beforeEach(function() {
    m=7;
    n=4;
    A = pool.zeros([m,n],'float64');
    v = pool.zeros([m*n-n*(n-1)/2],'float64');

    fill(A,function(i,j) { return Math.sqrt(2+i+j); });
  });

  it('returns a triangularized matrix in place of A',function() {
    householder.triangularize(A,v);
    assert( ndtest.matrixIsUpperTriangular(A,1e-8) );
  });

});

describe("multByQinv()", function() {

  var m,n,A,v,b;

  beforeEach(function() {
    m=7;
    n=4;
    A = pool.zeros([m,n],'float64');
    v = pool.zeros([m*n-n*(n-1)/2],'float64');
    b = ndarray([1,2,3,4,5,6,7]);

    fill(A,function(i,j) { return Math.sqrt(2+i+j); });
  });

  it('succeeds',function() {
    householder.triangularize(A,v);
    assert( householder.multByQinv(v,n,b) );
  });

});

describe("multByQ()", function() {

  var m,n,A,v,b;

  beforeEach(function() {
    m=7;
    n=4;
    A = pool.zeros([m,n],'float64');
    v = pool.zeros([m*n-n*(n-1)/2],'float64');
    b = ndarray([1,2,3,4,5,6,7]);

    fill(A,function(i,j) { return Math.sqrt(2+i+j); });
  });

  it('succeeds',function() {
    householder.triangularize(A,v);
    assert( householder.multByQ(v,n,b) );
  });

});

describe("constructQ()", function() {

  var m,n,A,v,b;

  beforeEach(function() {
    m=7;
    n=4;
    A = pool.zeros([m,n],'float64');
    v = pool.zeros([m*n-n*(n-1)/2],'float64');
    b = ndarray([1,2,3,4,5,6,7]);

    fill(A,function(i,j) { return Math.sqrt(2+i+j); });
  });

  it('succeeds',function() {
    householder.triangularize(A,v);
    assert( householder.multByQ(v,n,b) );
  });

});
