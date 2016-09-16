import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train): # loop through each training example
    scores = X[i].dot(W) # multiply 1 x 3073 * 3073, 10, get 10 dimensional vector
    correct_class_score = scores[y[i]] # index in the 10 dim vector
    for j in xrange(num_classes): # each score in index
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i, :].T
        dW[:, y[i]] -= X[i, :].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train


  # same for gradient
  dW /= num_train


  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # gradient of the regularization
  dW += reg * W # param plus the derivative of lambda* 1/2 * W**2 = lambda * W


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1.0
  scores = X.dot(W)
  correct_scores = scores[np.arange(num_train), y, None]
  margins = np.maximum(0, scores - correct_scores + delta)
  margins[np.arange(num_train), y] = 0 # set correct class margins to 0 (will be 1)
  loss = np.sum(margins)

  # Average loss
  loss /= num_train

  # add Reg
  loss += 0.5 * reg * np.sum(W * W) # add regularization

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins > 0] = 1 # if margins are greater than zero set them to 1
  margins[np.arange(num_train), y] = -1 * np.sum(margins, axis=1)
  dW = X.T.dot(margins)
  # Average over num trianing examples
  dW /= num_train

  # add reg gradient
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
