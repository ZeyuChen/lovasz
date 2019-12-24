from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import paddle.fluid as fluid

import numpy as np
from lovasz_losses import * 

def test_lovasz_grad():
    length = 5
    shape = [length]
    gt_sorted = fluid.layers.data(name='a', shape=shape, dtype='float32')
    gt_sorted = fluid.layers.reshape(gt_sorted, [-1, 1]) 
    jaccard = lovasz_grad(gt_sorted)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())
    
    data = np.arange(length, 0, -1).astype(np.float32)
    out = exe.run(test_program,
            fetch_list=[jaccard.name],
            feed = {'a':data}) 
    print(out[0], out[0].shape)

def test_lovasz_flatten_probas():
    probas_shape = [3, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    vprobas, vlabels = flatten_probas(probas, labels, ignore=ignore)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())
    
    probas_length = probas_shape[0] * probas_shape[1] * probas_shape[2]
    labels_length = labels_shape[0] * labels_shape[1] * labels_shape[2] 
    probas_data = np.arange(probas_length, 0, -1).reshape(probas_shape).astype(np.float32)

    values = np.array([2, 1, 1, 0])
    labels_data = values.reshape(labels_shape).astype(np.int32)
    ignore_data = np.array([1, 1, 1, 0], dtype=np.int32).reshape(labels_shape)
    probas_data = np.expand_dims(probas_data, 0)
    labels_data = np.expand_dims(labels_data, 0)
    ignore_data = np.expand_dims(ignore_data, 0)

    out1, out2 = exe.run(
            test_program,
            fetch_list=[vprobas.name, vlabels.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(out1, out1.shape)
    print(out2, out2.shape)

def test_lovasz_flatten_binary():
    probas_shape = [1, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    vprobas, vlabels = flatten_binary_scores(probas, labels, ignore=ignore)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())
    
    probas_length = probas_shape[0] * probas_shape[1] * probas_shape[2]
    labels_length = labels_shape[0] * labels_shape[1] * labels_shape[2] 
    probas_data = np.arange(probas_length, 0, -1).reshape(probas_shape).astype(np.float32)

    values = np.array([1, 1, 1, 0])
    labels_data = values.reshape(labels_shape).astype(np.int32)
    ignore_data = np.array([1, 1, 1, 0], dtype=np.int32).reshape(labels_shape)
    probas_data = np.expand_dims(probas_data, 0)
    labels_data = np.expand_dims(labels_data, 0)
    ignore_data = np.expand_dims(ignore_data, 0)

    out1, out2 = exe.run(
            test_program,
            fetch_list=[vprobas.name, vlabels.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(out1, out1.shape)
    print(out2, out2.shape)

def test_lovasz_hinge_all():
    probas_shape = [1, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    vprobas, vlabels = flatten_binary_scores(probas, labels, ignore=ignore)
    loss = lovasz_hinge_flat(vprobas, vlabels)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())

    probas_length = probas_shape[0] * probas_shape[1] * probas_shape[2]
    labels_length = labels_shape[0] * labels_shape[1] * labels_shape[2] 
    #probas_data = np.arange(probas_length, 0, -1).reshape(probas_shape).astype(np.float32)
    probas_data = np.array([0.23291497, 0.19052831, 0.79462721, 
        0.3297337]).reshape(probas_shape).astype(np.float32)

    values = np.array([1, 1, 1, 0])
    labels_data = values.reshape(labels_shape).astype(np.int32)
    ignore_data = np.array([1, 1, 1, 0], dtype=np.int32).reshape(labels_shape)
    probas_data = np.expand_dims(probas_data, 0)
    labels_data = np.expand_dims(labels_data, 0)
    ignore_data = np.expand_dims(ignore_data, 0)

    vp, vl, error = exe.run(
            test_program,
            fetch_list=[vprobas.name, vlabels.name, loss.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(vp)
    print(vl)
    print(error)

def test_lovasz_hinge_single():
    probas_shape = [1, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    lovasz_hinge(probas, labels, per_image=True, ignore=ignore)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())

    probas_length = probas_shape[0] * probas_shape[1] * probas_shape[2]
    labels_length = labels_shape[0] * labels_shape[1] * labels_shape[2] 
    probas_data = np.array([0.23291497, 0.19052831, 0.79462721, 
        0.3297337]).reshape(probas_shape).astype(np.float32)

    values = np.array([1, 1, 1, 0])
    labels_data = values.reshape(labels_shape).astype(np.int32)
    ignore_data = np.array([1, 1, 1, 0], dtype=np.int32).reshape(labels_shape)
    probas_data = np.expand_dims(probas_data, 0)
    labels_data = np.expand_dims(labels_data, 0)
    ignore_data = np.expand_dims(ignore_data, 0)

    loss = exe.run(
            test_program,
            fetch_list=[loss.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(loss)

def test_lovasz_softmax_flat():
    probas_shape = [3, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    vprobas, vlabels = flatten_probas(probas, labels, ignore=ignore)
#    loss = lovasz_softmax_flat(vprobas, vlabels, classes='present')
    loss = lovasz_softmax_flat(vprobas, vlabels, classes='all')
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())
    
    probas_length = probas_shape[0] * probas_shape[1] * probas_shape[2]
    labels_length = labels_shape[0] * labels_shape[1] * labels_shape[2] 

    probas_data = np.array([0.90709059, 0.08367204, 0.86837787, 0.45287163,
        0.83186626, 0.30239672, 0.45373512, 0.25180515, 0.92765359,
        0.05688091, 0.88769571, 0.31187094]).reshape(probas_shape).astype("float32")

    values = np.array([2, 1, 2, 0])
    labels_data = values.reshape(labels_shape).astype(np.int32)
    ignore_data = np.array([1, 1, 1, 0], dtype=np.int32).reshape(labels_shape)
    probas_data = np.expand_dims(probas_data, 0)
    labels_data = np.expand_dims(labels_data, 0)
    ignore_data = np.expand_dims(ignore_data, 0)
    vp, vl, loss = exe.run(
            test_program,
            fetch_list=[vprobas.name, vlabels.name, loss.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(vp)
    print(vl)
    print(loss)

def test_single_lovasz_softmax_flat():
    probas_shape = [1, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    vprobas, vlabels = flatten_probas(probas, labels, ignore=ignore)
    loss = lovasz_softmax_flat(vprobas, vlabels, classes='all')
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())
    
    probas_length = probas_shape[0] * probas_shape[1] * probas_shape[2]
    labels_length = labels_shape[0] * labels_shape[1] * labels_shape[2] 

    probas_data = np.array([0.90709059, 0.08367204, 0.86837787, 0.45287163]).reshape(probas_shape).astype("float32")

    values = np.array([1, 1, 1, 0])
    labels_data = values.reshape(labels_shape).astype(np.int32)
    ignore_data = np.array([0, 1, 1, 0], dtype=np.int32).reshape(labels_shape)
    probas_data = np.expand_dims(probas_data, 0)
    labels_data = np.expand_dims(labels_data, 0)
    ignore_data = np.expand_dims(ignore_data, 0)
    vp, vl, loss = exe.run(
            test_program,
            fetch_list=[vprobas.name, vlabels.name, loss.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(vp)
    print(vl)
    print(loss)

def test_lovasz_softmax():
    probas_shape = [3, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    loss = lovasz_softmax(probas, labels, classes='present', per_image=False,
            ignore=ignore)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())
    
    probas_data = \
            np.array([0.67192201, 0.81870198, 0.77258366, 0.14495502, 0.23867289,
                0.77602079, 0.6099467 , 0.84998262, 0.0811898 , 0.76430095,
                0.79137547, 0.99642811, 0.54362771, 0.16157255, 0.27041143,
                0.15576253, 0.02415075, 0.34646825, 0.87565063, 0.58807224,
                0.32440345, 0.66848973, 0.42985255, 0.67140429, 0.09327043,
                0.53752471, 0.11262969, 0.58534519, 0.22413638, 0.07070887,
                0.3450199 , 0.18854945, 0.77802942, 0.67194227, 0.80362883,
                0.59969173, 0.03151586, 0.74988883, 0.22059246, 0.58512586,
                0.16135984, 0.314945  , 0.97941178, 0.93608287, 0.43164755,
                0.94626785, 0.30692069, 0.95636061, 0.60257295, 0.38007489,
                0.38334655, 0.49893603, 0.35552424, 0.55916771, 0.90335988,
                0.64983527, 0.09720144, 0.02381892, 0.90885459, 0.35790411])

    probas_data = probas_data.reshape([5,3,2,2]).astype(np.float32)
    labels_data = np.array([1, 1, 2, 2, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 2, 2, 0,
        1, 2, 1], dtype='int32').reshape([5, 1, 2, 2]) 
    ignore_data = np.where(labels_data==0, 0, 1)
    loss = exe.run(
            test_program,
            fetch_list=[loss.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(loss)

def test_lovasz_hinge_all_v2():
    probas_shape = [1, 2, 2] 
    labels_shape = [2, 2, 1]
    probas = fluid.layers.data(name='p', shape=probas_shape, dtype='float32')
    labels = fluid.layers.data(name='l', shape=labels_shape, dtype='int32')
    ignore = fluid.layers.data(name='m', shape=labels_shape, dtype='int32')
    vprobas, vlabels = flatten_binary_scores(probas, labels, ignore=ignore)
    loss = lovasz_hinge_flat_v2(vprobas, vlabels)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program = fluid.default_main_program().clone(for_test=True)
    exe.run(fluid.default_startup_program())

    probas_length = probas_shape[0] * probas_shape[1] * probas_shape[2]
    labels_length = labels_shape[0] * labels_shape[1] * labels_shape[2] 
    probas_data = np.array([0.23291497, 0.19052831, 0.79462721, 
        0.3297337]).reshape(probas_shape).astype(np.float32)

    values = np.array([1, 1, 1, 0])
    labels_data = values.reshape(labels_shape).astype(np.int32)
    ignore_data = np.array([1, 1, 1, 0], dtype=np.int32).reshape(labels_shape)
    probas_data = np.expand_dims(probas_data, 0)
    labels_data = np.expand_dims(labels_data, 0)
    ignore_data = np.expand_dims(ignore_data, 0)

    vp, vl, error = exe.run(
            test_program,
            fetch_list=[vprobas.name, vlabels.name, loss.name],
            feed={'p':probas_data, 'l':labels_data, 'm':ignore_data})
    print(vp)
    print(vl)
    print(error)

if __name__ == '__main__':
    #test_lovasz_softmax()
    #test_lovasz_hinge_all()
    test_lovasz_hinge_all_v2()
    #test_lovasz_hinge_single()
    #test_single_lovasz_softmax_flat()
