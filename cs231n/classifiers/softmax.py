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
    num_train = X.shape[0]
    num_class = W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)
    f = np.dot(X,W)
    f -= np.max(f)
    for i in range(num_train):
        p = np.exp(f[i])/np.sum(np.exp(f[i]))
        for j in range(num_class):
            if j == y[i]:
                loss -= np.log(p[j])
                dW[:,j] += (p[j] -1) * X[i]
            else:
                dW[:,j] += p[j]*X[i]
    loss /= num_train
    loss += reg * np.sum(W*W)
    dW /= num_train
    dW += 2*reg*W
         
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)
    f = np.dot(X,W)
    f -= np.max(f)
    p_l = np.exp(f) / np.sum(np.exp(f), axis=1)[:,np.newaxis]
    loss = np.sum(-np.log(p_l[range(num_train),y]))/num_train +reg*np.sum(W*W)
    d_pl = p_l
    d_pl[range(num_train),y] -= 1
    dW = np.dot(X.T,d_pl)
    dW /= num_train
    dW += 2*reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    return loss, dW

