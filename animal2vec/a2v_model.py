#
# Analysis of animal-to-animal variability in IBL experiment
#
# model for animal-to-vector analysis
# 
import os, sys

import jax
import jax.numpy as jnp 	            # JAX NumPy
from jax import random, jit, grad

from flax import nnx  # The Flax NNX API.
from functools import partial

import optax

from one.api import ONE, OneAlyx
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

from a2v_data import generate_XY_data

import matplotlib.pyplot as plt


class MLPc(nnx.Module):
	def __init__(self, *, Na:int, Np:int, N0: int, N1: int, rngs: nnx.Rngs):
		self.linear1 = nnx.Linear(Na, N0, rngs=rngs)
		self.linear2 = nnx.Linear(N1, 32, rngs=rngs)
		self.linear3 = nnx.Linear(32, 2, rngs=rngs)

	def __call__(self, xt, xp, xa):
		x = jnp.concatenate( (xt, xp, self.linear1(xa)), axis=1 ) 
		x = nnx.relu(self.linear2(x))
		x = self.linear3(x)
		return x


def loss_fn_c(model: MLPc, Xt, Xa, Xp, Y, M):
	logits = model(Xt, Xa, Xp)
	loss = jnp.multiply( M, optax.softmax_cross_entropy_with_integer_labels(
		logits=logits, labels=Y) ).mean()
	return loss, logits


@nnx.jit
def train_step_c(model: MLPc, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, Xt, Xa, Xp, Y, M):
	"""Train for a single step."""
	grad_fn = nnx.value_and_grad(loss_fn_c, has_aux=True)
	(loss, logits), grads = grad_fn(model, Xt, Xa, Xp, Y, M)
	metrics.update(loss=loss, logits=logits, labels=Y)  # In-place updates.
	optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step_c(model: MLPc, metrics: nnx.MultiMetric, Xt, Xa, Xp, Y, M):
	loss, logits = loss_fn_c(model, Xt, Xa, Xp, Y, M)
	metrics.update(loss=loss, logits=logits, labels=Y)  # In-place updates.



class MLPcb(nnx.Module):
	def __init__(self, *, Na:int, N0: int, N1: int, rngs: nnx.Rngs):
		self.linear1 = nnx.Linear(Na, N0, rngs=rngs)
		self.linear2 = nnx.Linear(N1, 32, rngs=rngs)
		self.linear3 = nnx.Linear(32, 32, rngs=rngs)
		self.linear4 = nnx.Linear(32, 2, rngs=rngs)
		self.linear5 = nnx.Linear(32, 2, rngs=rngs)

	def __call__(self, xt, xp, xa):
		x = jnp.concatenate( (xt, xp, self.linear1(xa)), axis=1 )
		x = nnx.relu( self.linear2(x) )
		x = nnx.relu( self.linear3(x) )
		xc = self.linear4(x)
		xb = self.linear5(x)
		return xc, xb


def loss_fn_cb(model: MLPcb, Xt, Xa, Xp, Yc, Yb, M, cb_ratio, masked_session_length):
	logits, yhat_behav = model(Xt, Xa, Xp)
	loss_c = jnp.multiply( M, optax.softmax_cross_entropy_with_integer_labels(
		logits=logits, labels=Yc) ).mean() 
	loss_b = jnp.multiply( M, optax.squared_error(
		predictions=yhat_behav, targets=Yb).mean(axis=1) ).mean()
	#loss = loss_c + cb_ratio*loss_b
	loss = (loss_c + cb_ratio*loss_b)*(masked_session_length/400.0) # normalize loss by the session length
	return loss, [logits, loss_c, loss_b]


@nnx.jit
def train_step_cb(model: MLPcb, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, Xt, Xa, Xp, Yc, Yb, M, cb_ratio, mslen):
	"""Train for a single step."""
	grad_fn = nnx.value_and_grad(loss_fn_cb, has_aux=True)
	(loss, logits_loss_cb), grads = grad_fn(model, Xt, Xa, Xp, Yc, Yb, M, cb_ratio, mslen)
	c_accuracy = jnp.multiply( M, jnp.argmax(logits_loss_cb[0], axis=1) == Yc ).sum()/mslen
	metrics.update(total_loss=loss, behav_loss=logits_loss_cb[2], choice_accuracy=c_accuracy)  # In-place updates.
	optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step_cb(model: MLPcb, metrics: nnx.MultiMetric, Xt, Xa, Xp, Yc, Yb, M, cb_ratio, mslen):
	loss, logits_loss_cb = loss_fn_cb(model, Xt, Xa, Xp, Yc, Yb, M, cb_ratio, mslen)
	c_accuracy = jnp.multiply( M, jnp.argmax(logits_loss_cb[0], axis=1) == Yc ).sum()/mslen
	metrics.update(total_loss=loss, behav_loss=logits_loss_cb[2], choice_accuracy=c_accuracy)  # In-place updates.


	
