function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

 gradcalc = [-0.0092783
   0.0088991
  -0.0083601
   0.0076281
  -0.0067480
  -0.0055914
   0.0131540
   0.0197612
   0.0082794
  -0.0109273
  -0.0201749
  -0.0104983
   0.0081159
   0.0201475
   0.0126295
  -0.0058543
  -0.0191100
  -0.0151569
   0.0031508
   0.0180923
   0.3145450
   0.1110566
   0.0974007
   0.1489548
   0.0383952
   0.0448693
   0.1777077
   0.0775739
   0.0589954
   0.1474589
   0.0359237
   0.0384306
   0.1595309
   0.0735088
   0.0601514
   0.1438103
   0.0339263
   0.0315400];

  numgrad3 = [ -9.2783e-03
   8.8991e-03
  -8.3601e-03
   7.6281e-03
  -6.7480e-03
  -1.6768e-02
   3.9433e-02
   5.9336e-02
   2.4764e-02
  -3.2688e-02
  -6.0174e-02
  -3.1961e-02
   2.4923e-02
   5.9772e-02
   3.8641e-02
  -1.7370e-02
  -5.7566e-02
  -4.5196e-02
   9.1459e-03
   5.4610e-02
   3.1454e-01
   1.1106e-01
   9.7401e-02
   1.1868e-01
   3.8193e-05
   3.3693e-02
   2.0399e-01
   1.1715e-01
   7.5480e-02
   1.2570e-01
  -4.0759e-03
   1.6968e-02
   1.7634e-01
   1.1313e-01
   8.6163e-02
   1.3229e-01
  -4.5296e-03
   1.5005e-03];
                                   %NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


             % Setup some useful variables
%%Theta1(:,1) = 0;
%%Theta2(:,1) = 0
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);
% You need to return the following variables correctly 
D_2 = 0;
D_1 = 0;

       
        a_1 = [ones(m,1)  X];
        
        z_2 = a_1*Theta1';
        sz2 = sigmoid(z_2);
        a_2 = [ones(m,1) sz2];
        z_3 = a_2*Theta2';
        a_3 = sigmoid(z_3);
        h_x = a_3;
    
      
      for k=1 : num_labels
        
        
          
            y_k = y_matrix(:,k);
            
        
            hx_k = h_x(:,k);
            first = - y_k .*  log(hx_k );
            second = (1 - y_k) .* log(1 - hx_k);

        
            J =  J +  sum((sum( first - second)));
            
      
    
            
        end
                d_3 = a_3 - y_matrix;
                        
            d_2 = ((d_3* Theta2(:,2:end)) .* sigmoidGradient(z_2));
            
            
            D_1 = D_1 +    d_2'*a_1;
            
            D_2 = D_2 +    d_3'*a_2;
        
        J = J / (m);
        
gg= [ 0.76614;0.97990;0.37246;0.49749;0.64174;0.74614; ... 
0.88342;0.56876;0.58467;0.59814;1.92598;1.94462;1.98965;2.17855; ...
2.47834;2.50225;2.52644;2.72233];

ts1 = sum(sum((Theta1(:,2:end ).^ 2)));
ts2 = sum(sum((Theta2(:,2:end ).^ 2)));

regularization = (lambda / (2*m)) * (ts1 + ts2);
J = J +  regularization;

D_1_first = D_1(:,1);
D_1_rest = D_1(:,2:end);
mlayer = m;
lambda_dm = lambda/m;

Theta1_grad1 = D_1_first/mlayer;

regular_theta1 = lambda_dm * Theta1(:,2:end);

Theta1_grad = D_1*2/m;
%Theta1_grad  = [Theta1_grad1  regular_theta1 + (D_1_rest/mlayer)];


D_2_first = D_2(:,1);
D_2_rest = D_2(:, 2:end);
Theta2_grad1 = D_2_first;

regular_theta2 = lambda_dm * Theta2(:,2:end);

Theta2_grad = D_2*2/m;
%Theta2_grad = [Theta2_grad1   regular_theta2 + (D_2_rest/mlayer) ];


Theta1_grad_raveled = Theta1_grad / input_layer_size;

Theta2_grad_raveled = Theta2_grad / input_layer_size;
%gg1 = reshape(gg(1:6), 2,3);
%gg2 = reshape(gg(7:end), 4,3);

%gg1_i = gg1 * input_layer_size;
%gg2_i = gg2 * input_layer_size;

gg =gg *1;
%c1g = reshape( numgrad3(1:20), size(Theta1_grad));

%c2g = reshape( numgrad3(21:end), size(Theta2_grad));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)]/input_layer_size;

end
