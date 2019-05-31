## Modified by Pu for the ADMM l2 attack
## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

ITERATION_STEPS = 30  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-2  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
RO = 20
BINARY_SEARCH_STEPS = 9

class ADMML2:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 iteration_steps=ITERATION_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, ro=RO,
                 initial_const=INITIAL_CONST, binary_search_steps=BINARY_SEARCH_STEPS,
                 boxmin=-0.5, boxmax=0.5):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ITERATION_STEPS = iteration_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.ro = ro
        self.boxmin = boxmin
        self.boxmax = boxmax
        self.initial_const = initial_const
        self.BINARY_SEARCH_STEPS = binary_search_steps

        self.grad = self.gradient_descent(sess, model)

    def compare(self, x, y):
        if not isinstance(x, (float, int, np.int64)):
            x = np.copy(x)
            if self.TARGETED:
                x[y] -= self.CONFIDENCE
            else:
                x[y] += self.CONFIDENCE
            x = np.argmax(x)
        if self.TARGETED:
            return x == y
        else:
            return x != y

    def gradient_descent(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        # the variable we're going to optimize over
        #modifier = tf.Variable(tf.random_normal(shape, stddev=1.0))
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
        # these are variables to be more efficient in sending data to tf
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)
        const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        delta = tf.Variable(np.zeros(shape), dtype=tf.float32)
        multiplier = tf.Variable(np.zeros(shape), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_const = tf.placeholder(tf.float32, [batch_size])
        assign_delta = tf.placeholder(tf.float32, shape)
        assign_multiplier = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        boxmul = (self.boxmax - self.boxmin) / 2.
        boxplus = (self.boxmin + self.boxmax) / 2.
        newimg = tf.tanh(modifier + timg) * boxmul + boxplus

        modi_real = newimg - (tf.tanh(timg) * boxmul + boxplus)
        l2dist_real = tf.reduce_sum(tf.square(modi_real), [1, 2, 3])
        # prediction BEFORE-SOFTMAX of the model
        output = model.predict(newimg)

        # distance to the input data
        l2dist = tf.reduce_sum(tf.square(delta - modi_real + multiplier ),[1, 2, 3])

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((tlab) * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # sum up the losses
        loss2 = self.ro / 2.0 * l2dist
        loss1 = const*loss1
        loss_batch = loss1 + loss2
        loss = tf.reduce_sum(loss_batch)

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        init = tf.variables_initializer(var_list=new_vars)
        init1 = tf.variables_initializer(var_list=[modifier])

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(const.assign(assign_const))

        setup2 = []
        setup2.append(delta.assign(assign_delta))
        setup2.append(multiplier.assign(assign_multiplier))



        def doit(imgs, labs, delt, multi):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            lower_bound = np.zeros(batch_size)
            CONST = np.ones(batch_size) * self.initial_const
            upper_bound = np.ones(batch_size) * 1e10

            c_bestl2 = [1e10] * batch_size
            c_bestscore = [-1] * batch_size
            c_bestattack = [np.zeros(imgs[0].shape)] * batch_size

            bestlossbatch = [1e10] * batch_size
            modi = [np.zeros(imgs[0].shape)] * batch_size
            for binary_c_step in range(self.BINARY_SEARCH_STEPS):
      #          print('const:'+str(CONST))

                sess.run(init)
                sess.run(init1)
                bestl2 = [1e10] * batch_size
                bestscore = [-1] * batch_size

                sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_const: CONST, })

                sess.run(setup2, {assign_delta: delt, assign_multiplier: multi, })

                prev = 1e6

                for iteration in range(self.MAX_ITERATIONS):

                    sess.run([train])
                    l, lb, l2s, scores, nimg = sess.run([loss, loss_batch, l2dist_real, output, newimg])

        #            if iteration % (self.MAX_ITERATIONS // 10) == 0:
       #                 print(iteration, l)

                    # check if we should abort search if we're getting nowhere.
                    if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                        if l > prev * 1.09999:
                            break
                        prev = l

                    modi_re = sess.run(modi_real)

                    for e, (l2, lbb, sc, ii, mo) in enumerate(zip(l2s, lb, scores, nimg, modi_re)):
                        if l2 < bestl2[e] and self.compare(sc, np.argmax(batchlab[e])):
                            bestl2[e] = l2
                            bestscore[e] = np.argmax(sc)
                            #modi[e] = mo
                        if l2 < c_bestl2[e] and self.compare(sc, np.argmax(batchlab[e])):
                            c_bestl2[e] = l2
                            c_bestscore[e] = np.argmax(sc)
                            c_bestattack[e] = ii
                        if lbb < bestlossbatch[e]:
                            bestlossbatch[e] = lbb
                            modi[e] = mo

                for e in range(batch_size):
                    if self.compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                        # success, divide const by two
                        upper_bound[e] = min(upper_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        # failure, either multiply by 10 if no solution found yet
                        #          or do binary search with the known upper bound
                        lower_bound[e] = max(lower_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                        else:
                            CONST[e] *= 10

            return c_bestl2, c_bestscore, c_bestattack, modi

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r.extend(self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        boxmul = (self.boxmax - self.boxmin) / 2.
        boxplus = (self.boxmin + self.boxmax) / 2.
        simgs = np.arctanh((imgs - boxplus) / boxmul * 0.999999)

        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        multi = 0.0 * np.ones(imgs.shape)
        modi = 0.0 * np.ones(imgs.shape)

        for outer_step in range(self.ITERATION_STEPS):
            print(outer_step, o_bestl2)

        #    if outer_step == self.BINARY_SEARCH_STEPS // 2:
         #       self.LEARNING_RATE *= 0.1

            delt = self.ro / (self.ro + 2.0) * (modi - multi)

            bestl2, bestscore, bestattack, modi = self.grad(simgs, labs, delt, multi)

            multi = multi + delt - modi

            for e, (l2, sc, ii) in enumerate(zip(bestl2, bestscore, bestattack)):
                if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                    o_bestl2[e] = l2
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = ii

        return o_bestattack
