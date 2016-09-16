import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train, dim = X.shape
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      # compute vector of scores
      scores = X[i].dot(W)
      scores -= np.max(scores) # norm trick
      # scores are unnormed log probabilities, compute the probabilities
      sum_exp = 0.0
      for s in scores:
          sum_exp += np.exp(s)
      p = np.exp(scores) / sum_exp
      l_i = -np.log(p[y[i]])
      # add Loss for this training example to total loss
      loss += l_i

      # compute grad
      dldp = p
      dldp[y[i]] -= 1 # dcost with respect to scores, subtract one from correct class
      # fix dimms
      dldp = dldp.reshape(num_classes,1) # 10 x 1
      dpdW = X[i].reshape(dim, 1) # dscores with respect to weights 3073 x 1

      dW_i = np.dot(dpdW, dldp.T) # chain rule
      dW += dW_i





  # divide loss by num_examples to get average loss
  loss /= num_train
  dW /= num_train

  # add reg
  loss += reg * 0.5 * np.sum(W**2)
  dW += reg * W





  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # setup
  num_train, dim = X.shape
  num_classes = y.shape
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # forward pass
  scores = X.dot(W)
  scores -= np.max(scores) # norm trick
  # correct_scores = scores[np.arange(num_train), y]
  p = np.exp(scores) / np.reshape(np.sum(np.exp(scores),axis=1),(num_train,1)) # why does this work?
  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  # backward pass
  dldP = np.copy(p) # copy scores matrix 500 x 10 of normed probabilities
  # derivitave of loss func with respect to scores p
  dldP[np.arange(num_train), y] -= 1 # subtract 1 from all correct indexes
  # d of scores with respect to each weight (matrix multiply so just X)
  dPdW = X # 500 x 3073
  # chain rule to get dW (3073 x 10)
  dW = np.dot(dPdW.T, dldP)


  # average loss
  loss /= num_train
  #average gradient
  dW /= num_train

  # add L2 regularization
  loss += reg * 0.5 * np.sum(W ** 2)
  dW += reg * W # add reg to gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
