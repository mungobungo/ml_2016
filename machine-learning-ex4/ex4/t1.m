function t1 = t1()
  il = 2;              % input layer
  hl = 2;              % hidden layer
  nl = 4;              % number of labels
  nn = [ 1:18 ] / 10;  % nn_params
  X = cos([1 2 ; 3 4 ; 5 6]);
  y = [4; 2; 3];
  lambda = 0;
  [J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda)
end;

% test case
% >> [J grad] = nnCostFunction(nn, il, hl, nl, X, y, 0)
% J =  7.4070
% grad =
%   0.766138
%   0.979897
%  -0.027540
%  -0.035844
%  -0.024929
%  -0.053862
%   0.883417
%   0.568762
%   0.584668
%   0.598139
%   0.459314
%   0.344618
%   0.256313
%   0.311885
%   0.478337
%   0.368920
%   0.259771
%   0.322331