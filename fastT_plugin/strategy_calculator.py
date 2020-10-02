from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.labelled_graph import *
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io
import copy
import os
import time
import numpy as np
from queue import PriorityQueue as PQ


class Algorithm(object):
    def __init__(self, graph, nx_graph, nodes_must_in_cpu, cost_model, strategy_result, processor_name, batch_size,
                 split_prefix="strategy_split/", open_split=False):
        self._graph = graph
        self._nx_graph = nx_graph
        self._nodes_must_in_cpu = nodes_must_in_cpu
        self._reverse_graph = self._nx_graph.reverse()
        self.init_strategy_result = strategy_result
        self._cost_model = cost_model
        self._processor_name = processor_name
        self.cpu = '/device:CPU:0'
        self.master_gpu = '/job:localhost/replica:0/task:0/device:GPU:0'
        self.batch_size = batch_size
        self.split_prefix = split_prefix
        self._open_split = open_split
        self.memory_limit = 15 * 1024 * 1024 * 1024/6

    def compute_size_needed_for_compute_op(self, node, nx_graph):
        size = 0
        for input_edge in nx_graph.in_edges(nbunch=node, data=True):
            size += input_edge[2]["size"]

        return size

    def compute_size_needed_for_stateful_op(self, node, nx_graph):
        size = 0
        for output_edge in nx_graph.out_edges(nbunch=node, data=True):
            size += output_edge[2]["size"]

        return size

    def cp_priority(self, existing_place_strategy=None, nx_graph=None):
        if nx_graph == None:
            nx_graph = self._nx_graph
        rank_u = dict()
        rank_d = dict()
        _reverse_graph = nx_graph.reverse()

        print("computing rank_u ")
        import copy
        frozen_graph = copy.deepcopy(nx_graph)
        graph = nx.DiGraph(frozen_graph)
        # graph.remove_node('_SOURCE')
        reverse_graph = graph.reverse()
        exit_node = None
        for node in reverse_graph.nodes():
            if reverse_graph.in_degree(node) == 0:
                exit_node = node
                break
        assert (exit_node == "_SINK")
        # if existing_place_strategy:
        #    rank_u[exit_node] = self.node_cost_model(exit_node, existing_place_strategy[node],nx_graph)
        # else:
        #    rank_u[exit_node] = self.average_node_cost(exit_node,nx_graph)
        # reverse_graph.remove_node(exit_node)
        while len(reverse_graph.nodes()):
            waitting_to_delete = list()
            for node in reverse_graph.nodes():
                if reverse_graph.in_degree(node) == 0:
                    immediate_succs = self.get_immediate_successors(node, nx_graph)
                    max_value = 0

                    if existing_place_strategy:
                        for immediate_succ in immediate_succs:
                            assert immediate_succ in rank_u.keys()
                            memcpy_cost = self.memcpy_cost_model(
                                existing_place_strategy[node], existing_place_strategy[immediate_succ],
                                nx_graph[node][immediate_succ]['size'])
                            assert (memcpy_cost > 0)
                            max_value = max(max_value, rank_u[immediate_succ] + memcpy_cost)
                            assert (max_value > rank_u[immediate_succ])
                        rank_u[node] = self.node_cost_model(node, existing_place_strategy[node], nx_graph) + max_value
                    else:
                        for immediate_succ in immediate_succs:
                            assert immediate_succ in rank_u.keys()
                            memcpy_cost = self.average_edge_cost(
                                nx_graph[node][immediate_succ]['size'])
                            assert (memcpy_cost > 0)
                            max_value = max(max_value, rank_u[immediate_succ] + self.average_edge_cost(
                                nx_graph[node][immediate_succ]['size']))
                            assert (max_value > rank_u[immediate_succ])
                        rank_u[node] = self.average_node_cost(node, nx_graph) + max_value

                    waitting_to_delete.append(node)
            reverse_graph.remove_nodes_from(waitting_to_delete)

        print("computing rank_d ")

        frozen_graph = copy.deepcopy(nx_graph)
        graph = nx.DiGraph(frozen_graph)
        # graph.remove_node('_SOURCE')

        exit_node = None
        for node in graph.nodes():
            if graph.out_degree(node) == 0:
                exit_node = node
                break
        assert (exit_node == "_SINK")

        while len(graph.nodes()):
            waitting_to_delete = list()
            for node in graph.nodes():
                if graph.in_degree(node) == 0:
                    immediate_preds = self.get_immediate_preccessors(node, nx_graph)
                    max_value = 0
                    rank_d[node] = 0

                    if existing_place_strategy:
                        for immediate_pred in immediate_preds:
                            assert immediate_pred in rank_d.keys()
                            max_value = max(max_value,
                                            self.node_cost_model(immediate_pred,
                                                                 existing_place_strategy[immediate_pred],
                                                                 nx_graph) + rank_d[
                                                immediate_pred] + self.memcpy_cost_model(
                                                existing_place_strategy[immediate_pred],
                                                existing_place_strategy[node],
                                                nx_graph[immediate_pred][node]['size']))
                        rank_d[node] = max_value
                    else:
                        for immediate_pred in immediate_preds:
                            assert immediate_pred in rank_d.keys()
                            max_value = max(max_value, self.average_node_cost(immediate_pred, nx_graph) + rank_d[
                                immediate_pred] + self.average_edge_cost(nx_graph[immediate_pred][node]['size']))
                        rank_d[node] = max_value

                    waitting_to_delete.append(node)
            graph.remove_nodes_from(waitting_to_delete)

        assert (len(rank_d) == len(rank_u))
        priority = dict()
        for node, _rank_d in rank_d.items():
            priority[node] = _rank_d + rank_u[node]
            if node == "_SOURCE" or node == "_SINK":
                print("node name:%s, rank_d:%f,rank_u:%f,priority:%f" % (node, _rank_d, rank_u[node], priority[node]))

        return priority

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def get_nodes_in_critical_path(self, priority=None, existing_place_strategy=None, nx_graph=None):
        _start_time = time.time()
        if priority == None:
            priority = self.cp_priority(existing_place_strategy, nx_graph)
        ret = list()
        critical_path_priority = max(priority.values())
        print("priority of _SOURCE:%f,priority of _SINK:%f" % (priority["_SOURCE"], priority["_SINK"]))
        assert (self.isclose(priority["_SOURCE"], priority["_SINK"]))
        for node, _priority in priority.items():
            if self.isclose(critical_path_priority, _priority):
                ret.append(node)
        if existing_place_strategy:
            ret.sort(key=lambda node: self.node_cost_model(node, existing_place_strategy[node], nx_graph), reverse=True)
        else:
            ret.sort(key=lambda node: self.average_node_cost(node, nx_graph), reverse=True)
        # print("-------in get nodes in critical path------")
        # print("time spent:%f"%(time.time()-_start_time))
        return ret

    def default_order(self, nx_graph=None):
        if nx_graph == None:
            nx_graph = self._nx_graph
        schedule_order = list()
        import copy
        frozen_graph = copy.deepcopy(nx_graph)
        graph = nx.DiGraph(frozen_graph)
        # graph.remove_node('_SOURCE')
        while len(graph.nodes()):
            waitting_to_delete = list()
            for node in graph.nodes():
                if graph.in_degree(node) == 0:
                    waitting_to_delete.append(node)
                    schedule_order.append(node)
            graph.remove_nodes_from(waitting_to_delete)
        return schedule_order

    def memcpy_cost_model(self, send_device, recv_device, size):
        # return 0.1
        if send_device == recv_device:
            return 0.1
        key = send_device + recv_device
        if key not in self._cost_model['edge'].keys():
            return 0.1
        memcpy_model = self._cost_model['edge'][key]
        coeff = memcpy_model[0]
        incept = memcpy_model[1]
        _min_val = memcpy_model[2]
        if self.isclose(coeff, 0.0):
            return 0.1
        memcpy_time = size * coeff + incept
        if memcpy_time > 0:
            return memcpy_time
        else:
            return 0.1

    def node_cost_model(self, node, processor, nx_graph):
        split_num = 1
        origin_node_name = node
        if node == "_SINK" or node == "_SOURCE":
            return 0.1
        if node not in self._cost_model['node'].keys():
            if ("split_num" in nx_graph.node[node] and "origin_name" in nx_graph.node[node]):
                split_num = nx_graph.node[node]["split_num"]
                origin_node_name = nx_graph.node[node]["origin_name"]
                return 1.0 + (self.node_cost_model(origin_node_name, processor, nx_graph) / split_num)
            else:
                assert ("strategy_split" in node)
                return 1.0

        if (processor in self._cost_model['node'][origin_node_name].keys()):
            cost = self._cost_model['node'][origin_node_name][processor]
            if "CPU" in processor and node not in self._nodes_must_in_cpu:
                return 10000000
            if cost <= 0:
                assert (cost >= 0)
                return 0.1
            return (self._cost_model['node'][origin_node_name][processor])
        elif "CPU" in processor:
            if node in self._nodes_must_in_cpu:
                return 0.1
            else:
                return 10000000  # self.average_node_cost(node,nx_graph)
        else:
            return self.average_node_cost(node, nx_graph)

    def average_node_cost(self, node, nx_graph):
        average_cost = 0
        counter = 0
        origin_node_name = node
        if node not in self._cost_model['node'].keys():
            if ("split_num" in nx_graph.node[node] and "origin_name" in nx_graph.node[node]):
                split_num = nx_graph.node[node]["split_num"]
                origin_node_name = nx_graph.node[node]["origin_name"]
                return 1.0 + (self.average_node_cost(origin_node_name, nx_graph) / split_num)
            else:
                assert ("strategy_split" in node)
                return 1.0

        for processor, cost in self._cost_model['node'][origin_node_name].items():
            if "CPU" in processor:
                continue
            counter += 1
            average_cost += cost
        counter = counter if counter != 0 else 1
        return 1.0 + (average_cost / counter)

    def average_edge_cost(self, size):
        # return 0.1
        average_cost = 0
        for key, memcpy_model in self._cost_model['edge'].items():
            coeff = memcpy_model[0]
            incept = memcpy_model[1]
            _min_val = memcpy_model[2]
            memcpy_time = max(size * coeff + incept, _min_val)
            average_cost += memcpy_time
        if average_cost < 0:
            average_cost = 0
        total = len(self._cost_model['edge'])
        if total == 0:
            return 1.0
        return 1.0 + (average_cost / total)

    def get_immediate_successors(self, node, nx_graph=None):

        if nx_graph == None:
            nx_graph = self._nx_graph
        immediate_succs = list()
        for out_edge in nx_graph.out_edges(nbunch=node, data=True, default=False):
            immediate_succs.append(out_edge[1])
        return immediate_succs

    def get_immediate_preccessors(self, node, nx_graph=None):
        if nx_graph == None:
            nx_graph = self._nx_graph
        immediate_preccs = list()
        for input_edge in nx_graph.in_edges(nbunch=node, data=True, default=False):
            immediate_preccs.append(input_edge[0])
        return immediate_preccs

    def open_split(self):
        return self._open_split

    def compute(self, strategy=None, nx_graph=None):
        if strategy == None:
            strategy = self.init_strategy_result["_PLACE"]
        if nx_graph == None:
            nx_graph = self._nx_graph
        print("in compute")
        # nx_graph = self._nx_graph
        EFT, strategy, real_node_sequence = self._compute(strategy, nx_graph)
        print("EFT:%f" % EFT)

        strategy_result = (EFT, strategy, real_node_sequence)
        return strategy_result

    def compute_split(self, EFT, strategy, real_node_sequence):
        print("in compute split")
        nx_graph = self._nx_graph

        strategy_result = (EFT, strategy, real_node_sequence)

        critical_path = self.get_nodes_in_critical_path(priority=None, existing_place_strategy=strategy,
                                                        nx_graph=nx_graph)

        replace_list = None
        if self.open_split():
            replace_list = list()
            splited_nx_graph = nx_graph
            for i in range(2):
                unsuccess_time = 0
                success = False
                print("!!critical path:")
                print(critical_path)
                for node in critical_path:
                    # if "strategy_split" in node or "BatchNorm" in node or "train_op" in node or "_SINK"in node or "_SOURCE" in node or "BackpropFilter" in node or "c" in node:

                    print("node name in critical path:%s" % node)
                    if "strategy_split" not in node and "op_type" not in nx_graph.node[node]:
                        print("not in  node name:%s" % node)
                    if "_SINK" == node or "_SOURCE" == node or "strategy_split" in node or nx_graph.node[node][
                        "op_type"] != "Conv2DBackpropInput":
                        continue
                    worth, result = self.worth_split(node, EFT, strategy, splited_nx_graph)
                    if worth:
                        strategy_result = result[0]
                        split_num = result[1]
                        splited_nx_graph = result[2]
                        print("worth! split num:%d" % split_num)
                        # self.split(node, split_num, strategy)
                        replace_list.append((node, split_num, splited_nx_graph))
                        success = True
                        break
                    else:
                        unsuccess_time += 1
                        if unsuccess_time > 1:
                            break
                if success:
                    strategy = strategy_result[1]
                    EFT = strategy_result[0]
                    critical_path = self.get_nodes_in_critical_path(priority=None,
                                                                    existing_place_strategy=strategy,
                                                                    nx_graph=splited_nx_graph)
                    critical_path.sort(key=lambda node: self.node_cost_model(node, strategy[node], splited_nx_graph),
                                       reverse=True)
                else:
                    break

        strategy_result = strategy_result + (replace_list,)
        return strategy_result

    def split(self, node, split_num, strategy, nx_graph):
        for j in range(split_num):
            name = self.split_prefix + node + str(j)
            strategy[name] = self._processor_name[j + 1]
            nx_graph.add_node(name, split_num=split_num, origin_name=node,
                              batch_size=nx_graph.node[node]["batch_size"] / split_num,
                              is_stateful=nx_graph.node[node]["is_stateful"], op_type=nx_graph.node[node]["op_type"])
            preccs = self.get_immediate_preccessors(node, nx_graph)
            succs = self.get_immediate_successors(node, nx_graph)
            for pred in preccs:
                size = nx_graph[pred][node]["size"]
                nx_graph.add_edge(pred, name, size=size)
                assert ("op_type" in nx_graph.node[pred])
                assert ("op_type" in nx_graph.node[name])
            for succ in succs:
                size = nx_graph[node][succ]["size"]
                nx_graph.add_edge(name, succ, size=size)
                assert ("op_type" in nx_graph.node[name])
                assert ("op_type" in nx_graph.node[succ])
        nx_graph.remove_node(node)
        return strategy

    def worth_split(self, node, EFT, strategy, origin_nx_graph):
        print("in worth_split")
        print("node name: %s,execution time:%f, init EFT:%f" % (
        node, self.node_cost_model(node, strategy[node], origin_nx_graph), EFT))

        import copy
        nx_graph = copy.deepcopy(origin_nx_graph)
        processes = list(self._processor_name)
        processes.remove(processes[0])
        results = list()
        for i in range(2, len(processes) + 1):
            print("split = %d" % i)
            if self.batch_size % i != 0:
                continue
            updated_strategy = self.split(node, i, strategy, nx_graph)
            result = self._compute(updated_strategy, nx_graph)
            print("EFT under split %d :%f" % (i, result[0]))
            results.append((result, i, nx_graph))
            # self._nx_graph = copy.deepcopy(nx_graph)
            nx_graph = copy.deepcopy(origin_nx_graph)  # roll_back_graph
        EFTs = [result[0][0] for result in results]
        min_eft = min(EFTs)
        if min_eft >= EFT:
            return False, None
        else:
            index = EFTs.index(min_eft)
            result = results[index]
            return True, result

    def get_available_time_slot(self, aval, processor, pred_time, duration):
        # if "CPU" in processor:
        return pred_time
        if processor not in aval:
            import sys
            aval[processor] = list()
            aval[processor].append((0, sys.maxint))
            return 0
        else:
            for i in range(len(aval[processor])):
                _tuple = aval[processor][i]
                if _tuple[0] < pred_time and pred_time + duration < _tuple[1]:
                    return _tuple[0]
            return aval[processor][-1][0]

    def update_available_time_slot(self, aval, processor, EST, EFT):
        # if "CPU" in processor:
        return
        assert (processor in aval.keys())
        _tuple = None
        for start, end in aval[processor]:
            if start <= EST and EFT <= end:
                _tuple = (start, end)
                break
        assert (_tuple)
        start = _tuple[0]
        end = _tuple[1]
        index = aval[processor].index(_tuple)
        aval[processor][index] = (EFT, end)
        aval[processor].insert(index, (start, EST))

    def simulate(self, place_strategy, node_sequence, replace_list=None):
        EST = dict()
        EFT = dict()
        aval = dict()
        if replace_list and len(replace_list) > 0:
            nx_graph = replace_list[-1][2]
        else:
            nx_graph = self._nx_graph

        if node_sequence == None:
            node_sequence = self.default_order(nx_graph)
        assert (node_sequence[-1] == "_SINK")
        reverse_graph = nx_graph.reverse()
        EFT["_SOURCE"] = 0
        for node in node_sequence:
            # ret = nx.bfs_successors(reverse_graph, node)
            immediate_preds = self.get_immediate_preccessors(node, nx_graph)
            # print('immediat_preds')
            # print(immediate_preds)
            EST[node] = 0
            EFT[node] = self.node_cost_model(node, place_strategy[node], nx_graph)

            pred_cost = [0]
            for immed_pred in immediate_preds:
                pred_device = place_strategy[immed_pred]
                tensor_size = nx_graph[immed_pred][node]['size']
                memcpy_time = self.memcpy_cost_model(send_device=pred_device, recv_device=place_strategy[node],
                                                     size=tensor_size)
                pred_cost.append(EFT[immed_pred] + memcpy_time)
            execution_duration = self.node_cost_model(node, place_strategy[node], nx_graph)
            pred_dependency_earliest_start = max(pred_cost)
            EST[node] = max(
                self.get_available_time_slot(aval, place_strategy[node], pred_dependency_earliest_start,
                                             execution_duration),
                pred_dependency_earliest_start)
            EFT[node] = EST[node] + execution_duration

            self.update_available_time_slot(aval, place_strategy[node], EST[node],
                                            EFT[node])
        return EFT[node_sequence[-1]]

    def _compute(self, existing_place_strategy=None, nx_graph=None):
        raise RuntimeError('function _compute must be override by derived class')


class HEFT(Algorithm):
    def __init__(self, graph, nx_graph, nodes_must_in_cpu, cost_model, place_strategy, processor_name, batch_size,
                 open_split):
        Algorithm.__init__(self, graph, nx_graph, nodes_must_in_cpu, cost_model, place_strategy, processor_name,
                           batch_size, open_split=open_split)

    def heft_order(self, existing_place_strategy=None, nx_graph=None):
        if nx_graph == None:
            nx_graph = self._nx_graph

        rank = dict()
        import copy
        frozen_graph = copy.deepcopy(nx_graph)
        graph = nx.DiGraph(frozen_graph)
        # graph.remove_node('_SOURCE')
        reverse_graph = graph.reverse()
        exit_node = None
        for node in reverse_graph.nodes():
            if reverse_graph.in_degree(node) == 0:
                exit_node = node
                break
        assert (exit_node == "_SINK")
        # rank[exit_node] = 0
        # reverse_graph.remove_node(exit_node)
        while len(reverse_graph.nodes()):
            waitting_to_delete = list()
            for node in reverse_graph.nodes():
                if reverse_graph.in_degree(node) == 0:
                    # ret = nx.bfs_successors(nx_graph, node)
                    immediate_succs = self.get_immediate_successors(node, nx_graph)
                    max_value = 0
                    if existing_place_strategy:
                        for immediate_succ in immediate_succs:
                            assert immediate_succ in rank
                            memcpy_cost = self.memcpy_cost_model(
                                existing_place_strategy[node], existing_place_strategy[immediate_succ],
                                nx_graph[node][immediate_succ]['size'])
                            assert (memcpy_cost > 0)
                            max_value = max(max_value, rank[immediate_succ] + memcpy_cost)
                            assert (max_value > rank[immediate_succ])
                        rank[node] = self.node_cost_model(node, existing_place_strategy[node], nx_graph) + max_value
                    else:
                        for immediate_succ in immediate_succs:
                            assert immediate_succ in rank
                            memcpy_cost = self.average_edge_cost(
                                nx_graph[node][immediate_succ]['size'])
                            assert (memcpy_cost > 0)
                            max_value = max(max_value, rank[immediate_succ] + self.average_edge_cost(
                                nx_graph[node][immediate_succ]['size']))
                            assert (max_value > rank[immediate_succ])
                        rank[node] = self.average_node_cost(node, nx_graph) + max_value

                    waitting_to_delete.append(node)
            reverse_graph.remove_nodes_from(waitting_to_delete)

        schedule_order = [item[0] for item in sorted(rank.items(), key=lambda d: d[1], reverse=True)]
        # for node in schedule_order:
        print("max_rank:%f" % max(rank.values()))
        return schedule_order

    def get_schedule_order(self, existing_place_strategy=None, nx_graph=None):

        return self.heft_order(existing_place_strategy, nx_graph)

        # return self.default_order()

    def _compute(self, existing_place_strategy=None, nx_graph=None):

        must_in_cpu_counter = 0
        EST = dict()
        EFT = dict()
        aval = dict()
        place = dict()
        used_size = dict()
        used_size_for_stateful_op = dict()
        start_time = time.time()
        for processor in self._processor_name:
            used_size[processor] = 0
            used_size_for_stateful_op[processor] = 0

        if nx_graph == None:
            nx_graph = self._nx_graph
            reverse_graph = self._reverse_graph
        else:
            reverse_graph = nx_graph.reverse()

        # ret = nx.bfs_successors(nx_graph, "_SOURCE")
        node_sequence = self.get_schedule_order(existing_place_strategy, nx_graph)
        EST["_SOURCE"] = dict()
        EFT["_SOURCE"] = dict()
        place["_SOURCE"] = self.master_gpu
        for processor in self._processor_name:
            # aval[processor] = 0
            EST["_SOURCE"][processor] = 0
            EFT["_SOURCE"][processor] = 0

        # print("------heft----before big loop-------------")
        # print("time spent: %f"%(time.time()-start_time))
        start_time = time.time()
        for node in node_sequence:

            size_need_for_compute_op = 0
            size_need_for_stateful_op = 0
            if node != "_SOURCE" and node != "_SINK":
                if "is_stateful" not in nx_graph.node[node]:
                    print("!!!not have is stateful, op name:%s" % node)
                if nx_graph.node[node]["is_stateful"]:
                    size_need_for_stateful_op = self.compute_size_needed_for_stateful_op(node, nx_graph)
                else:
                    size_need_for_compute_op = self.compute_size_needed_for_compute_op(node, nx_graph)
            assert (node not in place or 'group_leader' in nx_graph.node[node] or node == "_SOURCE")

            # print("----heft--1-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()

            immediate_preds = self.get_immediate_preccessors(node, nx_graph)
            # print('immediat_preds')
            # print(immediate_preds)

            # print("----heft--2-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()
            EST[node] = dict()
            EFT[node] = dict()
            for processor in self._processor_name:
                pred_cost = [0]
                for immed_pred in immediate_preds:
                    pred_device = place[immed_pred]
                    tensor_size = nx_graph[immed_pred][node]['size']
                    memcpy_time = self.memcpy_cost_model(send_device=pred_device, recv_device=processor,
                                                         size=tensor_size)
                    pred_cost.append(EFT[immed_pred][pred_device] + memcpy_time)
                execution_duration = self.node_cost_model(node, processor, nx_graph)
                # if "GPU" in processor:
                #    print("requested memory:%f,used memory:%f,sum:%f" % (
                #    size_need_for_stateful_op+size_need_for_compute_op, used_size[processor], used_size[processor] + size_need_for_stateful_op+size_need_for_compute_op))
                if "GPU" in processor and size_need_for_stateful_op + size_need_for_compute_op + used_size[
                    processor] > self.memory_limit /1:
                    execution_duration = 1000 * execution_duration  # if exceed the memory limits, we assume the execution time is 10 times than usual
                    print("limited!!! requested memory:%f,used memory:%f,sum:%f"%(size_need_for_stateful_op+size_need_for_compute_op,used_size[processor],used_size[processor]+size_need_for_stateful_op+size_need_for_compute_op))
                pred_dependency_earliest_start = max(pred_cost)
                EST[node][processor] = max(
                    self.get_available_time_slot(aval, processor, pred_dependency_earliest_start, execution_duration),
                    pred_dependency_earliest_start)
                EFT[node][processor] = EST[node][processor] + execution_duration

            # print("----heft--3-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()

            if node in place:
                continue

            values = list(EFT[node].values())
            # print("values:%s"%values)
            index = values.index(min(values))
            # print("index:%d"%index)
            selected_processor = list(EFT[node].keys())[index]
            # print("EFT[node].keys():%s"%EFT[node].keys())

            #######group heuristic####

            #if len(immediate_preds) == 1:
            #    pred = immediate_preds[0]
            #    pred_succs = self.get_immediate_successors(pred, nx_graph)
            #    if len(pred_succs) == 1 and pred_succs[0] == pred:
             #       selected_processor = place[pred]

            #if 'colocation' in nx_graph.node[node]:
             #   colocation_node = nx_graph.node[node]['colocation']
                # print("colocation key:%s"%colocation_node)
                # ret = nx.bfs_successors(reverse_graph, colocation_node)
                # immediate_preds = ret.next()
                # print('immediat_preds of colocation node:')
                # print(immediate_preds)
            #    if place.has_key(colocation_node):
            #        selected_processor = place[colocation_node]
            #    else:
            #        place[colocation_node] = selected_processor

            if node in self._nodes_must_in_cpu:
                must_in_cpu_counter += 1
                selected_processor = selected_processor.replace("GPU","CPU",1)
                selected_processor = selected_processor[:-1]+"0"

            place[node] = selected_processor
            if "GPU" in selected_processor:
                used_size_for_stateful_op[selected_processor] += size_need_for_stateful_op
                used_size[selected_processor] = used_size_for_stateful_op[selected_processor] + size_need_for_compute_op if "gradient" not in node else used_size_for_stateful_op[selected_processor]
            # print("%s,%s"%(node,place[node]))
            # aval[selected_processor] = EFT[node][selected_processor]
            self.update_available_time_slot(aval, selected_processor, EST[node][selected_processor],
                                            EFT[node][selected_processor])

        # place.pop("_SOURCE")
        for key,value in used_size.items():
            print(key,value)

        real_start_time = dict()
        for node in node_sequence:
            device = place[node]
            real_start_time[node] = EST[node][device]

        real_node_sequence = [item[0] for item in sorted(real_start_time.items(), key=lambda d: d[1], reverse=False)]

        device_order_seqs = dict()
        order = dict()

        for node in real_node_sequence:
            device = place[node]
            if device not in device_order_seqs:
                device_order_seqs[device] = list()
            device_order_seqs[device].append(node)
        for device, node_seq in device_order_seqs.items():
            counter = 0
            for node in node_seq:
                order[node] = counter
                counter += 1

        print("finish _compute")
        print("num of node must in cpu:%d" % must_in_cpu_counter)
        return EFT["_SINK"][place[
            "_SINK"]], place, real_node_sequence


class CP(Algorithm):
    def __init__(self, graph, nx_graph, nodes_must_in_cpu, cost_model, place_strategy, processor_name, batch_size,
                 open_split):
        Algorithm.__init__(self, graph, nx_graph, nodes_must_in_cpu, cost_model, place_strategy, processor_name,
                           batch_size, open_split=open_split)

    def get_priority(self, existing_place_strategy=None, nx_graph=None):
        _start_time = time.time()

        ret = self.cp_priority(existing_place_strategy, nx_graph)

        # print("--------in get priority------")
        # print("time spent:%f"%(time.time()-_start_time))

        return ret

        # return self.default_order()

    def get_best_processor_for_critical_path(self, nodes_in_critical_path, nx_graph=None):
        costs = list()
        for device in self._processor_name:
            cost = 0
            for node in nodes_in_critical_path:
                cost += self.node_cost_model(node, device, nx_graph)
            costs.append(cost)
        min_index = costs.index(min(costs))
        return self._processor_name[min_index]

    def _compute(self, existing_place_strategy=None, nx_graph=None):

        start_time = time.time()
        if nx_graph == None:
            nx_graph = self._nx_graph
            reverse_graph = self._reverse_graph
        else:
            reverse_graph = nx_graph.reverse()
        must_in_cpu_counter = 0
        EST = dict()
        EFT = dict()
        aval = dict()
        place = dict()
        used_size = dict()
        used_size_for_stateful_op = dict()
        for processor in self._processor_name:
            used_size_for_stateful_op[processor] = 0
            used_size[processor] = 0

        priority = self.get_priority(existing_place_strategy, nx_graph)
        EST["_SOURCE"] = dict()
        EFT["_SOURCE"] = dict()
        place["_SOURCE"] = self.master_gpu
        for processor in self._processor_name:
            # aval[processor] = 0
            EST["_SOURCE"][processor] = 0
            EFT["_SOURCE"][processor] = 0

        nodes_in_critical_path = self.get_nodes_in_critical_path(priority, existing_place_strategy, nx_graph)
        best_processor = self.get_best_processor_for_critical_path(nodes_in_critical_path, nx_graph)
        import copy
        frozen_graph = copy.deepcopy(nx_graph)
        graph = nx.DiGraph(frozen_graph)
        # graph.remove_node('_SOURCE')

        priority_queue = PQ()
        for node in graph.nodes():
            if graph.in_degree(node) == 0:
                priority_queue.put((-priority[node], node))
        counter = 0

        # print("----cpop--before big loop-------------")
        # print("time spent: %f"%(time.time()-start_time))
        start_time = time.time()

        while priority_queue.empty() == False:

            node = priority_queue.get()[1]
            counter += 1
            size_need_for_compute_op = 0
            size_need_for_stateful_op = 0
            if node != "_SOURCE" and node != "_SINK":
                if nx_graph.node[node]["is_stateful"]:
                    size_need_for_stateful_op = self.compute_size_needed_for_stateful_op(node, nx_graph)
                else:
                    size_need_for_compute_op = self.compute_size_needed_for_compute_op(node, nx_graph)
            # print("counter:%d,node:%s"%(counter,node))
            assert (node not in place or 'group_leader' in nx_graph.node[node] or node == "_SOURCE")

            # print("----cpop--1-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()

            # ret = nx.bfs_successors(reverse_graph, node)
            immediate_preds = self.get_immediate_preccessors(node, nx_graph)
            # print('immediat_preds')
            # print(immediate_preds)
            EST[node] = dict()
            EFT[node] = dict()

            # print("----cpop--2-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()
            for processor in self._processor_name:
                pred_cost = [0]
                for immed_pred in immediate_preds:
                    pred_device = place[immed_pred]
                    tensor_size = nx_graph[immed_pred][node]['size']
                    memcpy_time = self.memcpy_cost_model(send_device=pred_device, recv_device=processor,
                                                         size=tensor_size)
                    pred_cost.append(EFT[immed_pred][pred_device] + memcpy_time)
                execution_duration = self.node_cost_model(node, processor, nx_graph)

                if "GPU" in processor and size_need_for_stateful_op + size_need_for_compute_op + used_size[
                    processor] > self.memory_limit / 1:
                    execution_duration = 1000 * execution_duration  # if exceed the memory limits, we assume the execution time is 10 times than usual
                    print("limit!!! requested memory:%f,used memory:%f,sum:%f"%(size_need_for_stateful_op+size_need_for_compute_op,used_size[processor],used_size[processor]+size_need_for_stateful_op+size_need_for_compute_op))
                pred_dependency_earliest_start = max(pred_cost)
                EST[node][processor] = max(
                    self.get_available_time_slot(aval, processor, pred_dependency_earliest_start, execution_duration),
                    pred_dependency_earliest_start)
                EFT[node][processor] = EST[node][processor] + execution_duration

                # if node =="InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/weights/RMSProp_1" or True:
                #    print("node name:%s"%node)
                #    print("processer:%s"%processor)
                #    print("pred_dependency_earliest_start:%f"%pred_dependency_earliest_start)
                #    print("node_cost:%s"%execution_duration)
                #    print("EST:%f" % EST[node][processor])
                #    print("EFT:%f"% EFT[node][processor])

            # print("----cpop----3-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()

            if node in place:
                immediate_succs = self.get_immediate_successors(node, graph)
                for immediate_succ in immediate_succs:
                    if graph.in_degree(immediate_succ) == 1:
                        priority_queue.put((-priority[immediate_succ], immediate_succ))
                graph.remove_node(node)
                continue

            values = list(EFT[node].values())
            # print("values:%s"%values)
            index = values.index(min(values))
            # print("index:%d"%index)
            selected_processor = list(EFT[node].keys())[index]
            # if node == "InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/weights/RMSProp_1" or True:
            #    print("selected processor:%s"%selected_processor)

            #if nx_graph.in_degree(node) == 1:
                # ret = nx.bfs_successors(reverse_graph, node)
            #    immediate_pred = self.get_immediate_preccessors(node, nx_graph)[0]
            #    if nx_graph.out_degree(immediate_pred) == 1:
             #       if 'CPU' not in place[immediate_pred]:
             #           selected_processor = place[immediate_pred]

            if node in nodes_in_critical_path:
                selected_processor = best_processor

            # if node == "InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/weights/RMSProp_1" or True:
            #    print("selected processor:%s"%selected_processor)

            # print("EFT[node].keys():%s"%EFT[node].keys())
            #if 'colocation' in nx_graph.node[node]:
            #    colocation_node = nx_graph.node[node]['colocation']
                # print("colocation key:%s"%colocation_node)
            #    if place.has_key(colocation_node):
            #        selected_processor = place[colocation_node]
            #    else:
            #        place[colocation_node] = selected_processor

            # if "Queue" in nx_graph.node[node]["op_type"]:
            #    selected_processor = self._processor_name[0]

            # if node == "InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/weights/RMSProp_1" or True:
            #    print("selected processor:%s"%selected_processor)
            if node in self._nodes_must_in_cpu:
                must_in_cpu_counter += 1
                selected_processor = selected_processor.replace("GPU","CPU",1)
                selected_processor = selected_processor[:-1]+"0"

            place[node] = selected_processor
            if "GPU" in selected_processor:
                used_size_for_stateful_op[selected_processor] += size_need_for_stateful_op
                used_size[selected_processor] = used_size_for_stateful_op[selected_processor] + size_need_for_compute_op if "gradient" not in node else used_size_for_stateful_op[selected_processor]

            # print("%s,%s"%(node,place[node]))
            # aval[selected_processor] = EFT[node][selected_processor]
            # print("----cpop----4-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()
            self.update_available_time_slot(aval, selected_processor, EST[node][selected_processor],
                                            EFT[node][selected_processor])
            # print("----cpop----5-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()

            immediate_succs = self.get_immediate_successors(node, graph)
            for immediate_succ in immediate_succs:
                if graph.in_degree(immediate_succ) == 1:
                    priority_queue.put((-priority[immediate_succ], immediate_succ))
            graph.remove_node(node)

            # print("----cpop--6-------------")
            # print("time spent: %f" % (time.time() - start_time))
            start_time = time.time()

        # print("------after big loop-------------")
        # print("time spent: %f"%(time.time()-start_time))
        start_time = time.time()
        for key,value in used_size.items():
            print(key,value)
        # place.pop("_SOURCE")
        print("counter:%d,len:%d" % (counter, len(place)))

        # assert (counter == len(place))
        real_start_time = dict()
        for node in nx_graph.nodes():
            if node == '_SOURCE':
                continue
            device = place[node]
            real_start_time[node] = EST[node][device]

        real_node_sequence = [item[0] for item in sorted(real_start_time.items(), key=lambda d: d[1], reverse=False)]

        device_order_seqs = dict()
        order = dict()

        for node in real_node_sequence:
            device = place[node]
            if device not in device_order_seqs:
                device_order_seqs[device] = list()
            device_order_seqs[device].append(node)
        for device, node_seq in device_order_seqs.items():
            counter = 0
            for node in node_seq:
                order[node] = counter
                counter += 1
        print("num of node must in cpu:%d" % must_in_cpu_counter)

        # print("------last process-------------")
        # print("time spent: %f"%(time.time()-start_time))

        return EFT['_SINK'][place["_SINK"]], place, real_node_sequence


class Placer(object):
    def __init__(self, graph, gpu_num, batch_size):
        # load cost model in basic session
        self._placed = False
        self.default_device = '/device:GPU:0'
        self.cpu = '/device:CPU:0'
        self.master_gpu = '/job:localhost/replica:0/task:0/device:GPU:0'
        self.gpu_num = gpu_num
        self.batch_size = batch_size
        self.makespan = list()
        self._graph = graph

        self._group_leaders = list()
        for op in self._graph.get_operations():
            #op._set_attr("_class", attr_value_pb2.AttrValue(s=""))
            if op._colocation_dict:
                colocation_op_name = list(op._colocation_dict.keys())[0]
                if colocation_op_name not in self._group_leaders:
                    self._group_leaders.append(colocation_op_name)
        print("group leader len:%d" % len(self._group_leaders))

        self._cost_model = None
        self._strategy_result = dict()
        self._strategy_result["_PLACE"] = dict()
        self._strategy_result["_REPLACE"] = list()
        self._strategy_result["_SEQUENCE"] = None
        self._strategy_result["_REAL_TIME"] = None
        self._strategy_result["_STABLE_COUNT"] = 0
        self._processor_name = ["/job:localhost/replica:0/task:0/device:CPU:0"]
        for i in range(gpu_num):
            device = "/job:localhost/replica:0/task:0/device:GPU:"+str(i)
            self._processor_name.append(device)

        self._load_cost_model()
        self._load_strategy_result()
        self._load_best_strategy_result()
        print("stable_count:%d" % self._strategy_result["_STABLE_COUNT"])
        self._need_profile = False if self._strategy_result["_STABLE_COUNT"] > 2 else True

        self.split_prefix = "strategy_split/"
        self._open_split = False  # True #if self._strategy_result["_STABLE_COUNT"]>5 else False

        with tf.name_scope(self.split_prefix) as scope:
            self._split_name_scope = scope

        self._create_split_scale_mask()

        self.start_time = 0
        self.end_time = 0
        self.counter = 0

        self.no_need = []
        for op in self._graph.get_operations():
            if op.device=="" or "CPU" in op.device:
                self.no_need.append(op.name)
        with open("no_need_node.json","w") as f:
            json.dump(self.no_need,f)

    def _create_split_scale_mask(self):
        self.scale_masks = dict()

    def _report_time_for_current_strategy(self, real_time):
        self._strategy_result["_REAL_TIME"] = real_time

        if self._best_strategy_result == None:
            self._best_strategy_result = copy.deepcopy(self._strategy_result)
        else:
            best_time = self._best_strategy_result["_REAL_TIME"]
            print("real time for current strategy:%f" % real_time)
            print("real time for best    strategy:%f" % best_time)
            if real_time < best_time:
                print("replace best strategy with current strategy")
                self._best_strategy_result = copy.deepcopy(self._strategy_result)
            # else:
            #    self._strategy_result = copy.deepcopy(self._best_strategy_result)
            #    self._strategy_result["_STABLE_COUNT"]+=1
            #    print("roll back last best strategy")

    # print("strategy after comparing with best strategy:")
    # self.print_strategy(self._strategy_result["_PLACE"])

    def get_split_scale_mask(self, num, axis):
        assert (axis >= 0 and axis <= 3)
        key = "num:" + str(num) + "axis" + str(axis)
        if key in self.scale_masks.keys():
            return self.scale_masks[key]
        else:
            if axis == 0:
                scale = tf.constant([num, 1, 1, 1])
            elif axis == 1:
                scale = tf.constant([1, num, 1, 1])
            elif axis == 2:
                scale = tf.constant([1, 1, num, 1])
            elif axis == 3:
                scale = tf.constant([1, 1, 1, num])
            self.scale_masks[key] = scale
            return scale

    def open_split(self):
        return self._open_split

    def _print_cost_model(self):
        for name, device_cost_dict in self._cost_model['node'].items():
            print(name)
            for device_name, cost in device_cost_dict.items():
                print("    device: %s  cost:%f" % (device_name, cost))

    def _print_edge_model(self):
        for edge, tuple in self._cost_model['edge'].items():
            print("edge:%s, coefficient:%f,inception:%f" % (edge, tuple[0], tuple[1]))

    def _create_subgraph_dfs(self, G, node):
        """ bidirection, O(1)"""

        edges = nx.dfs_successors(G.reverse(), node)
        nodes = []
        for k, v in edges.items():
            # print("root:%s"%k)
            # print(v)
            nodes.extend([k])
            nodes.extend(v)

        return nx.DiGraph(G.subgraph(nodes))

    def _process_colocation_node(self, nx_graph):
        for node in nx_graph.nodes():
            if node == "_SOURCE" or node == "_SINK":
                continue
            op = self._graph.get_operation_by_name(node)
            if op._colocation_dict:
                nx_graph.node[node]['colocation'] = list(op._colocation_dict.keys())[0]
                if nx_graph.has_node(list(op._colocation_dict.keys())[0]):
                    nx_graph.node[list(op._colocation_dict.keys())[0]]['group_leader'] = True
        return nx_graph

    def _post_process(self, strategy, replace_list=None):
        for op in self._graph.get_operations():
            # if 'queue' in op.name:
            #    strategy[op.name] = '/device:CPU:0'
            #    continue
            if op._colocation_dict:
                op_device = strategy[op.name] if op.name in strategy.keys() else op.device
                colocation_op = list(op._colocation_dict.keys())[0]
                colocation_device = strategy[list(op._colocation_dict.keys())[0]] if colocation_op in strategy.keys() else self._graph.get_operation_by_name(colocation_op).device
                if (op_device != colocation_device):
                    # assert (not strategy.has_key(op.name) or op.name in replace_list or colocation_op in replace_list)
                    strategy[op.name] = colocation_device
        return strategy

    def _check_colocation_node_and_complete_strategy(self, place_strategy):
        # complete the whole op in strategy:
        print("check_colocation_node_and_complete_strategy")
        for op in self._graph.get_operations():
            if op.name not in place_strategy.keys():
                # print(op.name)
                colocation_device = None
                if op._colocation_dict:
                    colocation_op_name = list(op._colocation_dict.keys())[0]
                    colocation_device = place_strategy[colocation_op_name] if colocation_op_name in place_strategy.keys() else self._real_device_placement[colocation_op_name]
                # else:
                # print(op.name)
                if colocation_device:
                    # print(op.name)
                    place_strategy[op.name] = colocation_device
                else:
                    # print(op.name)
                    place_strategy[op.name] = self._real_device_placement[op.name]  # op.device#self.default_device

            # place_strategy[op.name] = colocation_device if colocation_device else '/device:CPU:0' #op.device

        '''
        for node,device in place_strategy.iteritems():
          if node == "_SOURCE":
            continue
          op = self._graph.get_operation_by_name(node)
          if op._colocation_dict:
            assert(device==place_strategy[op._colocation_dict.keys()[0]])
        '''
        for op in self._graph.get_operations():
            if op._colocation_dict:
                op_device = place_strategy[op.name]
                for colocation_op_name in op._colocation_dict.keys():
                    colocation_device = place_strategy[colocation_op_name]
                    if (op.name == "global_step/Assign"):
                        print("op name: %s, op device: %s, colocation_name: %s, colocation device: %s" % (
                            op.name, op.device, colocation_op_name, colocation_device))
                    if (op_device != colocation_device):
                        print("op name: %s, op device: %s, colocation_name: %s, colocation device: %s" % (
                        op.name, op.device, colocation_op_name, colocation_device))
                        #assert (op_device == colocation_device)

    def _check_after_place(self):
        for op in self._graph.get_operations():
            if op._colocation_dict:
                op_device = op.device
                for colocation_node_name in op._colocation_dict.keys():
                    colocation_device = self._graph.get_operation_by_name(colocation_node_name).device
                    if (op_device != colocation_device):
                        print("op:%s, colocation op:%s, op_device:%s, colocation op device:%s" % (
                        op.name, colocation_node_name, op_device, colocation_device))
                        #assert (op_device == colocation_device)

    def _init_nx(self, ops_to_fetch):

        G = nx.DiGraph()
        operations = self._graph.get_operations()
        G.add_node("_SOURCE", batch_size=self.batch_size, is_stateful=False, op_type="_SOURCE")
        G.add_node("_SINK", batch_size=self.batch_size, is_stateful=False, op_type="_SINK")
        for op in operations:
            G.add_node(op.name, batch_size=self.batch_size, is_stateful=op.op_def.is_stateful, op_type=op.type)

        for op in operations:
            inputs = op.inputs  # data inputs
            control_inputs = op.control_inputs  # control inputs
            if len(inputs) == 0 and len(control_inputs) == 0:
                G.add_edge("_SOURCE", op.name, size=0)
                assert ("op_type" in G.node[op.name])
                continue
            for input_tensor in inputs:
                import operator
                dypye = input_tensor.dtype
                # print("dtype")
                # print(dypye)
                # print("size:%d"%dypye.size)
                if input_tensor.shape.dims == None:
                    size = 0
                elif len(input_tensor.shape.dims) == 0:
                    size = 0
                else:
                    if input_tensor.shape.is_fully_defined():
                        # print("fully defined tensor")
                        size = dypye.size * input_tensor.shape.num_elements()
                    else:
                        # print("not fully defined tensor: %s"%input_tensor.name)
                        # print(input_tensor.shape.as_list)
                        size = self.batch_size
                        for dim in input_tensor.shape.dims:
                            if dim.value:
                                size *= dim.value
                        size = dypye.size * size
                input_op = input_tensor.op
                if not(input_op.type=="NextIteration"):
                    G.add_edge(input_op.name, op.name, size=size)
                    assert ("op_type" in G.node[op.name])
                    assert ("op_type" in G.node[input_op.name])
            for control_input_op in control_inputs:
                # if "train_op" in op.name:
                # print("op name:%s"%op.name)
                # print("control input:%s"%control_input_op.name)
                # if "group_deps" in op.name:
                # print("op name:%s"%op.name)
                # print("control input:%s"%control_input_op.name)
                # print("add control inputs:%s-->%s"%(control_input_op.name,op.name))
                G.add_edge(control_input_op.name, op.name, size=0)
                assert ("op_type" in G.node[op.name])
                assert ("op_type" in G.node[control_input_op.name])

        for op in ops_to_fetch:
            G.add_edge(op.name, "_SINK", size=0)
            assert ("op_type" in G.node[op.name])

        nx_graph = self._create_subgraph_dfs(G, "_SINK")

        self._process_colocation_node(nx_graph)
        self.nx_graph = nx_graph
        # nx.draw(nx_graph,with_labels=False,node_size=10,pos=nx.random_layout(nx_graph))
        # plt.savefig("fig.pdf")

        print("init graph number of nodes")
        print(G.number_of_nodes())
        print("init graph number of edges")
        print(G.number_of_edges())
        print("sub graph number of nodes")
        print(self.nx_graph.number_of_nodes())
        print("sub graph number of edges")
        print(self.nx_graph.number_of_edges())
        try:
            edges = nx.algorithms.cycles.find_cycle(self.nx_graph)
            print("find cycle")
            print(edges)
        except:
            pass

    def _load_cost_model(self):
        import pickle
        import os
        file_name = "./cost_model.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fh:
                self._cost_model = pickle.load(fh)
                print("load cost model")
                print("len of cost model: %d" % len(self._cost_model['node']))
        else:
            return

    def _load_strategy_result(self):
        import pickle
        import os
        file_name = "./place_strategy.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fh:
                self._strategy_result = pickle.load(fh)
        else:
            self._strategy_result["_PLACE"] = self._default_place()
            return

    def _load_best_strategy_result(self):
        import pickle
        import os
        file_name = "./best_place_strategy.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fh:
                self._best_strategy_result = pickle.load(fh)
        else:
            self._best_strategy_result = None
            return

    def save_cost_model(self):
        import pickle
        file_name = "./cost_model.pkl"
        print("save cost model")
        print("len of cost model: %d" % len(self._cost_model['node']))
        with open(file_name, 'wb') as fh:
            pickle.dump(self._cost_model, fh)

    def save_place_strategy(self):
        import pickle
        file_name = "./place_strategy.pkl"
        with open(file_name, 'wb') as fh:
            pickle.dump(self._strategy_result, fh)

    def save_best_strategy_result(self):
        import pickle
        file_name = "./best_place_strategy.pkl"

        with open(file_name, 'wb') as fh:
            pickle.dump(self._best_strategy_result, fh)

    def check_node_model_different_ratio(self, node_model):
        print("node mode different ratio")
        for i in range(1, 20):
            node_count = 0
            total_count = 0
            for name, cost in node_model.items():
                processor = self._real_device_placement.get(name, self._graph.get_operation_by_name(name).device)
                if cost == 0 or name not in self._cost_model['node'].keys() or processor not in  self._cost_model['node'][
                    name].keys():
                    continue
                total_count += 1
                if cost >= 20 and abs(self._cost_model['node'][name][processor] - cost) / float(cost) > float(i) / 10:
                    # print("device:%s, node name:%s, old cost:%f, current cost:%f"%(processor,name,self._cost_model['node'][name][processor],cost))
                    node_count += 1
            # print("node_model_different_ratio larger than %f:%f" % (float(i) / 10, float(node_count) / total_count))

    def check_cpu_node_model_different_ratio(self, node_model):
        print("cpu node mode different ratio")
        for i in range(1, 20):
            node_count = 0
            total_count = 0
            for name, cost in node_model.items():
                processor = self._real_device_placement.get(name, self._graph.get_operation_by_name(name).device)
                if "CPU" not in processor or cost == 0 or name not in self._cost_model['node'].keys() or processor not in \
                self._cost_model['node'][name].keys():
                    continue
                total_count += 1
                if cost >= 20 and abs(self._cost_model['node'][name][processor] - cost) / float(cost) > float(i) / 10:
                    # print("device:%s, node name:%s, old cost:%f, current cost:%f"%(processor,name,self._cost_model['node'][name][processor],cost))
                    node_count += 1
            print("cpu node_model_different_ratio larger than %f:%f" % (float(i) / 10, float(node_count) / total_count if total_count!=0 else 1))
        print("total cpu node num: %d" % total_count)

    def check_gpu_node_model_different_ratio(self, node_model):
        print("gpu node mode different ratio")
        for i in range(1, 20):
            node_count = 0
            total_count = 0
            for name, cost in node_model.items():
                processor = self._real_device_placement.get(name, self._graph.get_operation_by_name(name).device)
                if "GPU" not in processor or cost == 0 or name not in self._cost_model['node'].keys() or processor not in \
                self._cost_model['node'][name].keys():
                    continue
                total_count += 1
                if cost >= 20 and abs(self._cost_model['node'][name][processor] - cost) / float(cost) > float(i) / 10:
                    # print("device:%s, node name:%s, old cost:%f, current cost:%f"%(processor,name,self._cost_model['node'][name][processor],cost))
                    node_count += 1
            print("gpu node_model_different_ratio larger than %f:%f" % (float(i) / 10, float(node_count) / total_count if total_count!=0 else 1))
        print("total gpu node num: %d" % total_count)

    def check_edge_model_different_ratio(self, edge_model):
        # print("edge mode different ratio")
        for i in range(1, 41):
            node_count = 0
            total_count = 0
            for key, tupl in edge_model.items():
                if key not in self._cost_model['edge'].keys():
                    continue
                total_count += 1
                # print("edge name:%s"%key)
                # print("previous coefficient:%f"%self._cost_model['edge'][key][0])
                # print("current coefficient:%f"%tupl[0])
                if abs(self._cost_model['edge'][key][0] - tupl[0]) / float(tupl[0]) > float(
                        i) / 10:  # or abs(self._cost_model['edge'][key][1] - tupl[1]) / float(tupl[1]) > float(i) / 10:
                    node_count += 1
            if total_count == 0:
                continue
            # print("edge_model_different_ratio larger than %f:%f" % (float(i) / 10, float(node_count) / total_count))

    def update_cost_model(self, run_metadata):
        print("update_cost_model")
        place_strategy = self._strategy_result["_PLACE"]
        self.lgraph = LabelledGraph(graph=self._graph, run_metadata=run_metadata)
        name_cost_dict = self.lgraph.calculate_op_cost()
        print("calculate op cost finish")
        device_speed = self.lgraph.analysis_memcpy()
        print("analysis memcpy finish")
        if self._cost_model == None:
            self._cost_model = dict()
            self._cost_model['node'] = dict()
            for name, device_cost in name_cost_dict.items():
                if name in self._real_device_placement.keys():
                    device = self._real_device_placement[name]
                else:
                    device = self._graph.get_operation_by_name(name).device
                # print("device:%s"%device)
                self._cost_model['node'][name] = dict()
                self._cost_model['node'][name][device] = device_cost
            for name, device_cost in name_cost_dict.items():  # process default situation
                if self._graph.get_operation_by_name(name).device == "":
                    device = ""
                    self._cost_model['node'][name][device] = device_cost

            self._cost_model['edge'] = device_speed
        else:
            self.check_node_model_different_ratio(name_cost_dict)
            self.check_cpu_node_model_different_ratio(name_cost_dict)
            self.check_gpu_node_model_different_ratio(name_cost_dict)
            self.check_edge_model_different_ratio(device_speed)
            for name, device_cost in name_cost_dict.items():
                if name in self._real_device_placement.keys():
                    processor = self._real_device_placement[name]
                else:
                    processor = place_strategy.get(name, self._graph.get_operation_by_name(name).device)
                if name not in  self._cost_model['node'].keys():
                    self._cost_model['node'][name] = dict()
                if processor in self._cost_model['node'][name].keys():
                    self._cost_model['node'][name][processor] = self._cost_model['node'][name][processor] * 0.5 + \
                                                                device_cost * 0.5
                    # self._cost_model['node'][name][processor] = device_cost[1]
                else:
                    self._cost_model['node'][name][processor] = device_cost

                if self._graph.get_operation_by_name(name).device == "":  # process default situation
                    device = ""
                    self._cost_model['node'][name][device] = device_cost

                for device, tup in device_speed.items():
                    if device not in self._cost_model['edge'].keys():
                        self._cost_model['edge'][device] = tup
                    else:
                        coeffi = tup[0] * 0.5 + self._cost_model['edge'][device][0] * 0.5
                        incept = tup[1] * 0.5 + self._cost_model['edge'][device][1] * 0.5
                        _min_val = tup[2] * 0.5 + self._cost_model['edge'][device][2] * 0.5
                        # coeffi = tup[0]
                        # incept = tup[1]
                        self._cost_model['edge'][device] = (coeffi, incept, _min_val)

        return

    def _create_algorithm(self, algorithm):
        if algorithm == "heft":
            ret = HEFT(self._graph, self.nx_graph, self._nodes_must_in_cpu, self._cost_model, self._strategy_result,
                       self._processor_name,
                       self.batch_size, self._open_split)
        if algorithm == "cp":
            ret = CP(self._graph, self.nx_graph, self._nodes_must_in_cpu, self._cost_model, self._strategy_result,
                     self._processor_name,
                     self.batch_size, self._open_split)
        return ret

    def _algorithm_compute(self, algorithm):
        eft, strategy, real_node_sequence = algorithm.compute()
        strategy = self._post_process(strategy)
        return eft, strategy, real_node_sequence

    def _algorithm_split(self, algorithm, eft, strategy, real_node_sequence):
        eft, strategy, real_node_sequence, replace_list = algorithm.compute_split(eft, strategy, real_node_sequence)
        replace_node_name_list = None
        if replace_list:
            replace_node_name_list = [item[0] for item in replace_list]
        strategy = self._post_process(strategy, replace_node_name_list)
        return eft, strategy, real_node_sequence, replace_list

    def _simulate_algorithm(self, algorithm, place_strategy, sequence, replace_list=None):
        eft = algorithm.simulate(place_strategy, sequence, replace_list)
        return eft

    def _default_place(self):

        print("default place!!")

        op_device = dict()
        no_need = list()

        '''
        partition = (len(self._graph.get_operations())//self.gpu_num)+1
        for i,op in enumerate(self._graph.get_operations()):
            index = (i//partition)+1
            op_device[op.name] = self._processor_name[index]

        '''
        for op in self._graph.get_operations():
            op_device[op.name] = op.device



        # counter=0
        # for op in self._graph.get_operations():
        #    if not op_device.has_key(op.name):
        #       op_device[op.name] = self._processor_name[counter+1]
        #       counter = (counter+1)%self.gpu_num

        op_device["_SINK"] = self.default_device
        op_device["_SOURCE"] = self.default_device

        op_device = self._post_process(op_device)

        return op_device

    def _different_ratio(self, new_place_strategy):
        _place_strategy = self._strategy_result["_PLACE"]
        if _place_strategy == None:
            return 100
        counter = 0
        for name, device in new_place_strategy.items():
            if name not in  _place_strategy or device != _place_strategy[name]:
                counter += 1
        print("different ratio")
        print(float(counter) / len(new_place_strategy))
        return float(counter) / len(new_place_strategy)

    def have_dependency(self, pr_op, suc_op):
        inputs = suc_op.inputs
        for input_tensor in inputs:
            input_op = input_tensor.op
            if input_op.name == pr_op.name:
                return True
        control_inputs = suc_op.control_inputs
        for control_inputs_op in control_inputs:
            if control_inputs_op.name == pr_op.name:
                return True
        return False

    def split_inputs(self, op, num, batch_size):
        ret = list()
        status = False
        inputs = op.inputs
        op_type = op.type
        axis = 0

        for i in range(num):
            ret.append(list())

        if "Conv2DBackpropFilter" == op_type:
            if inputs[0].shape[3] % num != 0:  # split from in_channel dm.
                return False, None, None
            for i, input_ in enumerate(inputs):
                if i == 0:
                    print("can split, tensor name:" + input_.name)
                    print(input_.shape)
                    status = True
                    out = tf.split(input_, axis=3, num_or_size_splits=num)
                    print("after split, the new out shape:")
                    print(out[0].shape)

                elif i == 1:
                    print("can not split, tensor name:" + input_.name)
                    print(input_.shape)

                    scale = self.get_split_scale_mask(num, 2)
                    print("scale op name:%s" % scale.op.name)
                    print("scale's shape:")
                    print(scale.shape)
                    new_input = tf.floordiv(input_, scale)
                    print("new input shape:")
                    print(new_input.shape)
                    print("new_input op name:%s" % new_input.op.name)
                    out = [new_input] * num
                elif i == 2:
                    print("can not split, tensor name:" + input_.name)
                    print(input_.shape)
                    out = [input_] * num
                for j in range(num):
                    ret[j].append(out[j])
            axis = 2

        elif "Conv2DBackpropInput" == op_type:
            if inputs[2].shape[0] % num != 0:
                return False, None, None
            for i, input_ in enumerate(inputs):
                if i == 0:

                    scale = self.get_split_scale_mask(num, 0)
                    print("scale op name:%s" % scale.op.name)
                    print("scale's shape:")
                    print(scale.shape)
                    new_input = tf.floordiv(input_, scale)
                    print("new input shape:")
                    print(new_input.shape)
                    print("new_input op name:%s" % new_input.op.name)
                    out = [new_input] * num

                elif i == 1:
                    print("can not split, tensor name:" + input_.name)
                    print(input_.shape)
                    out = [input_] * num

                elif i == 2:
                    print("can split, tensor name:" + input_.name)
                    print(input_.shape)
                    status = True
                    out = tf.split(input_, axis=0, num_or_size_splits=num)
                    print("after split, the new out shape:")
                    print(out[0].shape)

                    print("can not split, tensor name:" + input_.name)
                    print(input_.shape)

                for j in range(num):
                    ret[j].append(out[j])
            axis = 0

        else:
            for input_ in inputs:
                if len(input_.shape) and input_.shape[0] == batch_size:  # 32 is batchsize
                    print("can split, tensor name:" + input_.name)
                    print(input_.shape)
                    status = True
                    out = tf.split(input_, axis=axis, num_or_size_splits=num)
                else:
                    print("can not split, tensor name:" + input_.name)
                    print(input_.shape)
                    out = [input_] * num
                for i in range(num):
                    ret[i].append(out[i])
        return status, ret, axis

    def find_input_index(self, op, input):
        inputs = op.inputs
        for i in range(len(inputs)):
            if input.name == inputs[i].name:
                return i
        return -1

    def activate_replace_list(self, replace_list, place_strategy):

        status = True
        need_finalized = False
        try:
            self._graph._check_not_finalized()
        except:
            self._graph._unsafe_unfinalize()
            need_finalized = True

        with tf.name_scope(self._split_name_scope):
            for node, split_num, _ in replace_list:
                status = status and self.replace(self._graph.get_operation_by_name(node), split_num, self.batch_size,
                                                 place_strategy)

        if need_finalized:
            self._graph.finalize()
        return status

    def replace(self, replace_op, num, batch_size, place_strategy):
        # return
        if batch_size % num != 0:
            return False
        status, ret, axis = self.split_inputs(replace_op, num, batch_size)
        if (status == False):
            return False
        print("replace node:%s" % replace_op.name)

        clone_ops = list()
        for i, inputs in enumerate(ret):
            _name = replace_op.name + str(i)
            device = place_strategy[self.split_prefix + _name]
            with tf.device(device):
                print("new inputs' shape")
                for _input in inputs:
                    print(_input.shape)
                node_def = node_def_pb2.NodeDef()
                node_def.CopyFrom(replace_op.node_def)
                node_def.attr["_class"].list.s[0] = ""
                op_clone = self._graph.create_op(replace_op.type,
                                                 inputs,
                                                 [out_t.dtype for out_t in replace_op.outputs],
                                                 name=_name,
                                                 op_def=replace_op.op_def,
                                                 attrs=node_def.attr)
            print("clone op's colocation dict")
            print(op_clone._colocation_dict)
            clone_ops.append(op_clone)
        for i in range(len(replace_op.outputs)):
            old_output = replace_op.outputs[i]
            print("old output shape:")
            print(old_output.shape)
            consumers = old_output.consumers()
            clone_outputs = [ops.outputs[i] for ops in clone_ops]
            print("new output shape before concate:")
            print(clone_outputs[0].shape)
            output = tf.concat(clone_outputs, axis=axis)
            print("concate output shape:")
            print(output.shape)
            for consumer in consumers:
                print("  consumer name:%s" % consumer.name)
                index = self.find_input_index(consumer, old_output)
                print("  index:%d" % index)
                assert (index != -1)
                consumer._update_input(index, output)
            consumers = old_output.consumers()
            assert (len(consumers) == 0)
        return True

    def print_strategy(self, place_strategy):
        device_dict = dict()
        for name, device in place_strategy.items():
            if name.startswith("_"):
                continue
            if device not in device_dict.keys():
                device_dict[device] = 0
            device_dict[device] += 1
        print("strategy used:")
        for device, counter in device_dict.items():
            print("device:%s,num:%d" % (device, counter))

    def activate_place(self):
        place_strategy = self._strategy_result["_PLACE"]
        replace_list = self._strategy_result["_REPLACE"]
        real_node_sequence = self._strategy_result["_SEQUENCE"]
        order = dict()
        if real_node_sequence:
            for i, item in enumerate(real_node_sequence):
                order[item] = i

        # print("replace list:")
        # print(replace_list)
        # for node in replace_list:
        #  self.replace(self._graph.get_operation_by_name(node),self.gpu_num,self.batch_size)
        # self._graph.finalize()

        self.activate_replace_list(replace_list, place_strategy)

        self.print_strategy(place_strategy)

        assert (place_strategy)
        for op in self._graph.get_operations():
            device = place_strategy[op.name]
            # with ops.device(device):
            #   self._graph._apply_device_functions(op)
            if op.name in self.no_need:
                continue
            op._set_device(device)
            op._set_attr("_priority", attr_value_pb2.AttrValue(i=order.get(op.name, 10000000)))

        # for name, device in place_strategy.iteritems():
        #  op = self._graph.get_operation_by_name(name)
        #  op._set_device(device)

        # op = self._graph.get_operation_by_name("softmax_cross_entropy_loss/xentropy")
        # self.replace(op,4)
        # op = self._graph.get_operation_by_name("aux_loss/xentropy")
        # self.replace(op, 4)
        # self._graph.finalize()

        # graphdef = self._graph.as_graph_def()
        # graph_without_senquence_control = tf.Graph()
        # with graph_without_senquence_control.as_default():
        #     tf.import_graph_def(graphdef, name="")
        #     graph_without_senquence_control = tf.get_default_graph()
        # sgv = ge.sgv(self._graph)

        # ge.copy(sgv,graph_without_senquence_control)

        #  op = graph_without_senquence_control.get_operation_by_name("global_step/Assign")
        #  op1 = self._graph.get_operation_by_name("global_step/Assign")
        #  assert(op._colocation_dict and op1._colocation_dict)

        '''
        print("begin to add control input!")

        if len(real_node_sequence)>2:
            counter=0
            loop = int((len(real_node_sequence)-1))
            loop = 160
            for i in range(loop):
                pr_name = real_node_sequence[i]
                suc_name = real_node_sequence[i+1]
                if(pr_name=="_SOURCE" or suc_name=="_SINK"):
                    continue
                pr_op = self._graph.get_operation_by_name(pr_name)
                suc_op = self._graph.get_operation_by_name(suc_name)
                if not self.have_dependency(pr_op,suc_op):
                    print("add control: %s------>%s"%(pr_name,suc_name))
                    suc_op._add_control_input(pr_op)
                    suc_op._control_flow_post_processing()
                    counter+=1
            print("num of control input added: %d\n"%(counter))
        '''

        #  op._set_attr('_priority',attr_value_pb2.AttrValue(i=priority))

        self._check_after_place()

    # self._graph = graph_without_senquence_control  #we need to compute new strategy without the control inputs added by last strategy

    def add_split_nodes_in_nx_graph(self, nx_graph):
        operations = self._graph.get_operations()
        for op in operations:
            if op.name not in nx_graph.nodes():
                nx_graph.add_node(op.name, is_stateful=op.op_def.is_stateful, op_type=op.type)

        for op in operations:
            inputs = op.inputs  # data inputs
            control_inputs = op.control_inputs  # control inputs
            if len(inputs) == 0 and len(control_inputs) == 0 and not nx_graph.has_edge("_SOURCE", op.name):
                nx_graph.add_edge("_SOURCE", op.name, size=0)
                assert ("op_type" in nx_graph.node[op.name])
                continue
            for input_tensor in inputs:
                dypye = input_tensor.dtype
                if input_tensor.shape.dims == None:
                    size = 0
                elif len(input_tensor.shape.dims) == 0:
                    size = 0
                else:
                    if input_tensor.shape.is_fully_defined():
                        size = dypye.size * input_tensor.shape.num_elements()
                    else:
                        size = self.batch_size
                        for dim in input_tensor.shape.dims:
                            if dim.value:
                                size *= dim.value
                        size = dypye.size * size
                input_op = input_tensor.op
                if not nx_graph.has_edge(input_op.name, op.name):
                    nx_graph.add_edge(input_op.name, op.name, size=size)
                    assert ("op_type" in nx_graph.node[op.name])
                    assert ("op_type" in nx_graph.node[input_op.name])
            for control_input_op in control_inputs:
                if not nx_graph.has_edge(control_input_op.name, op.name):
                    nx_graph.add_edge(control_input_op.name, op.name, size=0)
                    assert ("op_type" in nx_graph.node[op.name])
                    assert ("op_type" in nx_graph.node[control_input_op.name])

        nx_graph = self._create_subgraph_dfs(nx_graph, "_SINK")

        nx_graph = self._process_colocation_node(nx_graph)
        return nx_graph

    def _read_nodes_must_in_cpu(self):

        file_name = "./nodes_must_in_cpu.log"
        assert (os.path.exists(file_name))

        with open(file_name, 'r') as fh:
            line = fh.read()
        self._nodes_must_in_cpu = line.split(",")
        self._nodes_must_in_cpu.remove(self._nodes_must_in_cpu[-1])
        # print(self._nodes_must_in_cpu)

    def _read_real_device_placement(self):
        file_name = "./real_device_assignment.log"
        assert (os.path.exists(file_name))
        self._real_device_placement = dict()
        with open(file_name, 'r') as fh:
            for line in fh.readlines():
                line = line.strip()
                tmp = line.split(",")
                node = tmp[0]
                #device = '/' + tmp[1].split('/')[-1]
                device = tmp[1]
                self._real_device_placement[node] = device

        # for node,device in self._real_device_placement.iteritems():
        #   print("node:"+node)
        #  print("device:"+device)

    def _update_place_strategy(self):
        for node, device in self._strategy_result["_PLACE"].items():
            if device == "":
                assert node in self._real_device_placement.keys()
                real_device = self._real_device_placement[node]
                self._strategy_result["_PLACE"][node] = real_device

    def calculate_place_strategy(self, ops_to_fetch):

        self._init_nx(ops_to_fetch)

        self._read_nodes_must_in_cpu()

        stable = True

        algorithm1 = self._create_algorithm("cp")
        eft1, place_strategy1, real_node_sequence1 = self._algorithm_compute(algorithm1)
        algorithm2 = self._create_algorithm("heft")
        eft2, place_strategy2, real_node_sequence2 = self._algorithm_compute(algorithm2)
        # place_strategy = self._default_place()
        print("eft of cpop:%f" % eft1)
        print("eft of heft:%f" % eft2)
        if eft1 < eft2:
            print("using cpop")
            algorithm = algorithm1
            eft = eft1
            place_strategy = place_strategy1
            real_node_sequence = real_node_sequence1
        else:
            print("using heft")
            algorithm = algorithm2
            eft = eft2
            place_strategy = place_strategy2
            real_node_sequence = real_node_sequence2

        _start_time = time.time()

        eft_for_current_strategy = self._simulate_algorithm(algorithm, place_strategy, real_node_sequence)
        eft_for_old_strategy = self._simulate_algorithm(algorithm, self._best_strategy_result['_PLACE'],
                                                        self._best_strategy_result['_SEQUENCE'])

        # print("-----simulate time------")
        # print("time spent:%f"%(time.time()-_start_time))
        print("before split: eft_for_current_strategy:%f,eft_for_old_best_strategy:%f" % (
        eft_for_current_strategy, eft_for_old_strategy))
        if True:#eft_for_current_strategy <= eft_for_old_strategy:
            self._strategy_result['_PLACE'] = place_strategy
            self._strategy_result['_SEQUENCE'] = real_node_sequence
            stable = False

        if self._open_split:
            eft, place_strategy, real_node_sequence, replace_list = self._algorithm_split(algorithm, eft,
                                                                                          place_strategy,
                                                                                          real_node_sequence)
            if replace_list and len(replace_list) > 0:
                eft_for_current_strategy = self._simulate_algorithm(algorithm, place_strategy, real_node_sequence,
                                                                    replace_list)
                print("after split: eft_for_current_strategy:%f,eft_for_old_strategy:%f" % (
                    eft_for_current_strategy, eft_for_old_strategy))
                if eft_for_current_strategy < eft_for_old_strategy:
                    if self.activate_replace_list(replace_list, place_strategy):
                        self._check_colocation_node_and_complete_strategy(place_strategy)
                        new_nx_graph = self.add_split_nodes_in_nx_graph(replace_list[-1][-1])
                        EFT, place_strategy, real_node_sequence = algorithm.compute(place_strategy, new_nx_graph)
                        replace_node_name_list = [item[0] for item in replace_list]
                        place_strategy = self._post_process(place_strategy, replace_node_name_list)
                        self._strategy_result['_REPLACE'].extend(replace_list)
                        self._strategy_result['_PLACE'] = place_strategy
                        self._strategy_result['_SEQUENCE'] = real_node_sequence
                        stable = False

        for node, device in place_strategy.items():
            if self.split_prefix in node:
                print("node name:%s, device:%s" % (node, device))

        # else:
        #    self._strategy_result['_REPLACE'] = list()

        if stable:
            self._strategy_result["_STABLE_COUNT"] += 1
            if self._strategy_result["_STABLE_COUNT"] > 2:
                print("stable count larger than 2!!")
                self._strategy_result = copy.deepcopy(self._best_strategy_result)  # try to use best strategy
                self._best_strategy_result["_STABLE_COUNT"] += 1
                # self._strategy_result["_STABLE_COUNT"]=6
                # pass #to do: to choose the best strategy.

        else:
            self._strategy_result["_STABLE_COUNT"] = 0

        '''

        if eft_for_current_strategy<eft_for_old_strategy:
            if self._open_split:
                if replace_list:
                    if self.activate_replace_list(replace_list,place_strategy):
                        self._check_colocation_node_and_complete_strategy(place_strategy)
                        new_nx_graph = self.add_split_nodes_in_nx_graph(replace_list[-1][-1])
                        EFT, place_strategy, real_node_sequence=algorithm._compute(place_strategy,new_nx_graph)
                        self._strategy_result['_REPLACE'].extend(replace_list)
            else:
                self._strategy_result['_REPLACE'] = list()
            self._strategy_result['_PLACE'] = place_strategy
            self._strategy_result['_SEQUENCE'] = real_node_sequence
            self._strategy_result['_STABLE_COUNT'] = 0
        else:
            self._strategy_result['_STABLE_COUNT'] += self._strategy_result['_STABLE_COUNT']

        '''

        '''

        if eft_for_current_strategy<eft_for_old_strategy or algorithm.open_split():
            print("update strategy result")
            self._strategy_result['_PLACE'] = place_strategy
            self._strategy_result['_SEQUENCE'] = real_node_sequence

            # self._graph._unsafe_unfinalize()
            # for node in replace_list:
            #  print("replace node name:%s"%node)
            #  self.replace(self._graph.get_operation_by_name(node),self.gpu_num,self.batch_size)
            # self._graph.finalize()

            # if len(self._strategy_result['_REPLACE']) < 0:
            if algorithm.open_split():
                self.activate_replace_list(replace_list)
                self._check_colocation_node_and_complete_strategy()
                new_nx_graph = self.add_split_nodes_in_nx_graph(replace_list[-1][-1])
                EFT, strategy, real_node_sequence=algorithm._compute( self._strategy_result['_PLACE'],new_nx_graph)

                self._strategy_result['_REPLACE'].extend(replace_list)
            else:
                self._strategy_result['_REPLACE']=list()
        '''

        print("check before")
        # self.print_strategy()
        self.print_strategy(self._strategy_result['_PLACE'])

        self._check_colocation_node_and_complete_strategy(self._strategy_result['_PLACE'])
        print("check after")
        self.print_strategy(self._strategy_result['_PLACE'])

    def need_profile(self):
        #return False
        return self._need_profile