## Modified by Pu for the ADMM l0 attack
## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 30  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
RO = 20
ITERATION_STEPS = 30
INCEPTION = False

class ADMML0:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, in_ro = RO,
                 iteration_steps = ITERATION_STEPS, inception = INCEPTION,
                 boxmin=-0.5, boxmax=0.5):

        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.initial_ro = in_ro
        self.boxmin = boxmin
        self.boxmax = boxmax
        self.inception = inception
        self.ITERATION_STEPS = iteration_steps
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
        delta = tf.Variable(np.zeros(shape), dtype=tf.float32)
        multiplier = tf.Variable(np.zeros(shape), dtype=tf.float32)
        ro = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_delta = tf.placeholder(tf.float32, shape)
        assign_multiplier = tf.placeholder(tf.float32, shape)
        assign_ro = tf.placeholder(tf.float32, [batch_size])

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        boxmul = (self.boxmax - self.boxmin) / 2.
        boxplus = (self.boxmin + self.boxmax) / 2.
        newimg = tf.tanh(modifier + timg) * boxmul + boxplus
        oriimg = tf.tanh(timg) * boxmul + boxplus

        modi_real = newimg - (tf.tanh(timg) * boxmul + boxplus)

        # prediction BEFORE-SOFTMAX of the model
        output = model.predict(newimg)
        #output2 = model.predict(delta+tf.tanh(timg) * boxmul + boxplus)
        # distance to the input data
        l2dist = tf.reduce_sum(tf.square(delta - modi_real + multiplier),[1, 2, 3])

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
        loss2 = ro / 2.0 * l2dist
        loss1 = 200.0*loss1
        loss_batch = loss1 + loss2
        loss = tf.reduce_sum(loss_batch)

        l2dist_real = tf.reduce_sum(tf.square(modi_real), [1, 2, 3])

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

        setup2 = []
        setup2.append(delta.assign(assign_delta))
        setup2.append(multiplier.assign(assign_multiplier))
        setup2.append(ro.assign(assign_ro))

        sess.run(init1)
        def doit(imgs, labs, delt, multi, Ro):

            sess.run(init)
           # sess.run(init1)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]


            bestl0 = [1e10] * batch_size
            bestscorel0 = [-1] * batch_size
            bestattackl0 = [np.zeros(imgs[0].shape)] * batch_size

            sess.run(setup, {assign_timg: batch,
                             assign_tlab: batchlab, })

            sess.run(setup2, {assign_delta: delt,
                              assign_multiplier: multi,
                              assign_ro: Ro})

            prev = 1e6
            bestlossbatch = [1e10] * batch_size
            modi = [np.zeros(imgs[0].shape)] * batch_size

            oimg, _ = sess.run([oriimg, newimg])
            for iteration in range(self.MAX_ITERATIONS):

                sess.run([train])
                l, lb, l2s = sess.run([loss, loss_batch, l2dist_real])
                scores, nimg = sess.run([output, newimg])
             #   if iteration % (self.MAX_ITERATIONS // 10) == 0:
             #       print(iteration, l)

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * 1.09999:
                        break
                    prev = l

                modi_re = sess.run(modi_real)
                ros = sess.run([ro])

                roo = np.reshape(ros, [-1, 1, 1, 1])
    #            roo = roo * np.ones(imgs.shape)
                tempmo = np.where(np.abs(modi_re) ** 2 < (2.0 / roo), 0, modi_re)

                scors = np.zeros(batchlab.shape)
                if self.inception:
                    for i in range(batch_size):
                        scors[i]=(np.reshape(model.model.predict(oimg[i] + tempmo[i]), (batchlab[0:1].shape)))
                else:
                    scors = model.model.predict(oimg + tempmo)

                for e, (lbb, scs, tmo, mo) in enumerate(zip(lb, scors, tempmo, modi_re)):
                    l0s = np.array(np.nonzero(tempmo[e])).shape[1]
                    if l0s < bestl0[e] and self.compare(scs, np.argmax(batchlab[e])):
                        bestl0[e] = l0s
                        bestscorel0[e] = scs
                        bestattackl0[e] = tmo
                    if lbb < bestlossbatch[e]:
                        bestlossbatch[e] = lbb
                        modi[e] = mo

            return bestl0, bestscorel0, bestattackl0, modi

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
        o_bestl0 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = []
        #o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        for i in range(batch_size):
            o_bestattack.append(imgs[i])


        prero = self.initial_ro
        count1 = 0

        while count1 < 2:
            multi = 0.0 * np.ones(imgs.shape)
            modi = 0.0 * np.ones(imgs.shape)

            lower_bound = np.zeros(batch_size)
            RO = np.array( np.ones(batch_size) * self.initial_ro )
            upper_bound = np.ones(batch_size) * 1e10

            for binary_c_step in range(self.BINARY_SEARCH_STEPS):

            #multi = 0.0 * np.ones(imgs.shape)
            #modi = 0.0 * np.ones(imgs.shape)

                bestl0tp=[1e10] * batch_size
                bestscoretp=[-1] * batch_size
                bestattacktp=[]
                for i in range(batch_size):
                    bestattacktp.append(imgs[i])

                print(binary_c_step, RO)
                for outer_step in range(self.ITERATION_STEPS):
                    print(outer_step, o_bestl0)

                    temp = (modi - multi)

                    Roo = np.reshape(RO, [-1,1,1,1])
                    delt = np.where(temp**2 < (2.0/Roo), 0, temp)

                    #print(np.array(np.nonzero(delt)).shape[1])

                    bestl0, bestscore, bestattack, modi = self.grad(simgs, labs, delt, multi, RO)

                    multi = multi + delt - modi

                    for e, (l0, sc, ii) in enumerate(zip(bestl0, bestscore, bestattack)):
                        if l0 < bestl0tp[e] and self.compare(sc, np.argmax(labs[e])):
                            bestl0tp[e] = l0
                            bestscoretp[e] = np.argmax(sc)
                            bestattacktp[e] = ii + imgs[e]
                        if l0 < o_bestl0[e] and self.compare(sc, np.argmax(labs[e])):
                            o_bestl0[e] = l0
                            o_bestscore[e] = np.argmax(sc)
                            o_bestattack[e] = ii+imgs[e]

                for e in range(batch_size):
                    if self.compare(bestscoretp[e], np.argmax(labs[e])) and bestscoretp[e] != -1:
                        # success, divide const by two
                        upper_bound[e] = min(upper_bound[e], RO[e])
                        if upper_bound[e] < 1e9:
                            RO[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        # failure, either multiply by 10 if no solution found yet
                        #          or do binary search with the known upper bound
                        lower_bound[e] = max(lower_bound[e], RO[e])
                        if upper_bound[e] < 1e9:
                            RO[e] = (lower_bound[e] + upper_bound[e]) / 2
                        else:
                            RO[e] *= 9

            count1 += 1
            self.initial_ro *= 3

        self.initial_ro = prero
        return o_bestattack
