import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################


  score = X.dot(W)
  N = score.shape[0]
  C = score.shape[1]
  # print(score)
  score = np.exp(score)
  # print(score)
  sum = np.sum(score,axis=1)
  # print(sum)
  # prob = np.zeros_like(score)
  # for i in range(N):
  #   prob[i] = score[i] / sum[i]
  # print(prob)

  prob = np.zeros_like(score) #(N,C), i.e.(500,10)
  for i in range(N):
    loss += -np.log(score[i][y[i]] / sum[i])
    prob[i] = score[i] / sum[i]
    for j in range(C):
      if j == y[i]:
        dW[:,j] += (prob[i][j] - 1) * X[i].T
      else:
        dW[:,j] += prob[i][j] * X[i].T
  # print(dW)
  loss /= N
  loss += reg * np.sum(W*W)
  dW /= N
  dW += 2 * reg * W

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
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  score = np.exp(score)
  # print("score")
  # print(score)
  # print(np.sum(score,axis = 1).shape)
  sum = np.sum(score, axis=1).reshape(-1, 1)
  # print(sum.shape)
  # print(sum_col_vector)
  prob = score / sum
  # print("prob")
  # print(prob)
  prob_correct_class = prob[range(N), list(y)]
  # print("prob_correct_class")
  # print(prob_correct_class.shape)
  loss = np.sum(-np.log(prob_correct_class))
  loss = loss / N + np.sum(W*W)

  prob[range(N), list(y)] -= 1
  dW = (X.T).dot(prob)
  dW = dW / N + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


