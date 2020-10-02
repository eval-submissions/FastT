import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.framework import tensor_description_pb2
from tensorflow.python.framework import dtypes
from google.protobuf import text_format as pbtf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops

import json
import re
from sklearn import linear_model
import numpy as np
from scipy import stats

import networkx as nx

_COMMUNICATION_OPS = [
    "_Recv", "_HostRecv", "_Send", "_HostSend"
]


class LabelledGraph(object):
    def __init__(self,
                 run_metadata_path=None,
                 run_metadata=None,
                 graphdef_path=None,
                 graph=None):
        if run_metadata:
            self.run_metadata = run_metadata
        else:
            self.run_metadata = self._import_run_metadata(run_metadata_path)
        if graph:
            self.graph = graph
            self.graphdef = graph.as_graph_def()
        else:
            self.graphdef = self._import_graphdef(graphdef_path)
            g_1 = tf.Graph()
            with g_1.as_default():
                tf.import_graph_def(self.graphdef, name="")
                self.graph = tf.get_default_graph()

        self.partition_graphs = (self.run_metadata.partition_graphs if self.run_metadata else None)
        self.step_stats = (self.run_metadata.step_stats if self.run_metadata else None)
        self.cost_graph = (self.run_metadata.cost_graph if self.run_metadata else None)
        self.partition_graphs_devices = self._get_device_per_pgraph()
        self.partition_name_node_dict = self._get_partition_name_node_dict()
        self.graph_name_node_dict = self._get_graph_name_node_dict()
        self._batch_size = self._get_batch_size()

    def _get_batch_size(self):
        for dev_step_stat in self.step_stats.dev_stats:
            for step_stat in dev_step_stat.node_stats:
                if step_stat.node_name == "fifo_queue_Dequeue":
                    return step_stat.output[0].tensor_description.shape.dim[0].size

    def _get_name_cost_dict(self):
        assert (self.cost_graph != None)
        name_cost_dict = dict()
        for node in self.cost_graph.node:
            name_cost_dict[node.name] = node.compute_cost
        return name_cost_dict

    def _get_nx(self):
        G = nx.DiGraph()
        tf_graph = self.graph
        operations = tf_graph.get_operations()
        for op in operations:
            G.add_node(op._id)
        for op in operations:
            inputs = op.inputs
            for input_tensor in inputs:
                input_op = input_tensor.op
                G.add_edge(input_op._id, op._id)
        return G

    def _generate_labelled_graph_from_pgraph(self):
        labelled_graph = graph_pb2.GraphDef()
        for graph in self.partition_graphs:
            labelled_graph.node.extend(graph.node)

        return labelled_graph

    def get_labelled_graph(self):
        labelled_graph = self._generate_labelled_graph_from_pgraph()
        return labelled_graph

    def _import_run_metadata(self, filename=None):

        if not filename:
            return None
        '''
        bucket_name = "jitprofiling"
        host = "cn-hangzhou.oss.aliyun-inc.com"
        auth = oss2.AnonymousAuth()
        bucket_client = oss2.Bucket(auth, host, bucket_name)
        path = "run_metadata/"+filename
        remote_stream = bucket_client.get_object(path)
        print("Run Metadata Download Done.")
        content = remote_stream.read()
        print("Read Done.")
        run_metadata = config_pb2.RunMetadata()
        run_metadata.ParseFromString(content)
        with open('run_metadata'+filename, 'w') as f:
          f.write(str(run_metadata))
        '''
        with open(filename, 'r') as fh:
            content = fh.read()
        run_metadata = config_pb2.RunMetadata()
        # run_metadata.ParseFromString(content)
        pbtf.Parse(content, run_metadata)
        return run_metadata

    def _import_step_stats(self, filename=None):
        with open(filename, 'r') as fh:
            step_stats_str = fh.read()
        step_stats = step_stats_pb2.StepStats()
        pbtf.Parse(step_stats_str, step_stats)

        return step_stats

    def _import_partition_graphs(self, filename=None):
        with open(filename, 'r') as fh:
            partition_graphs_str = fh.read()

        partition_graphs_str = partition_graphs_str.strip('[]')
        partition_graphs = list()
        for graph_str in partition_graphs_str.split(","):
            gdef = graph_pb2.GraphDef()
            pbtf.Parse(graph_str, gdef)
            partition_graphs.append(gdef)
        return partition_graphs

    def _parse_tensor_description(self, tensor_description_str):

        tensor_description = step_stats_pb2.TensorDescription()
        pbtf.Parse(tensor_description_str, tensor_description)
        return tensor_description

    def _get_device_per_pgraph(self):
        devices = list()

        def get_device_info(gdef):
            for node in gdef.node:
                if node.device:
                    return node.device
            return None

        for gdef in self.partition_graphs:
            device = get_device_info(gdef)
            if not device:
                raise ValueError('Device in partition_graphs not found')
            elif device in devices:
                raise ValueError('Duplicate device in partition_graphs')
            devices.append(device)
        return devices

    def _import_timeline(self, filename=None):
        with open(filename, 'r') as fh:
            timeline = json.load(fh)
        timeline = timeline['traceEvents']
        return timeline

    def _import_graphdef(self, filename=None):
        if not filename:
            return None
        '''
        bucket_name = "jitprofiling"
        host = "cn-hangzhou.oss.aliyun-inc.com"
        auth = oss2.AnonymousAuth()
        bucket_client = oss2.Bucket(auth, host, bucket_name)
        path = "graph_def/" +filename
        remote_stream = bucket_client.get_object(path)
        print("Graph Def Download Done.")
        content = remote_stream.read()
        print("Read Done.")
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(content)
        print("Parse Done.")
        if graph_def is not None:
          with open('graph_def'+filename, 'w') as f:
            f.write(str(graph_def)) 
        '''
        graph_def = graph_pb2.GraphDef()
        with open(filename, 'r') as fh:
            content = fh.read()
        # graph_def.ParseFromString(content)
        pbtf.Parse(content, graph_def)
        return graph_def

    def _get_raw_name_from_step_stats(self, name):
        suffix = None
        raw_name = name
        if re.match(r"^edge_[0-9]+_*", name):
            prefix = re.findall("edge_[0-9]+_", name)[0]
            raw_name = raw_name[len(prefix):]

        splits = raw_name.split(':', 1)
        raw_name = splits[0]
        if len(splits) == 2:
            suffix = splits[1]

        return raw_name, suffix

    def _get_pre_and_raw_name_from_step_stats(self, name):
        raw_name = name
        suffix = None
        splits = raw_name.split(':', 1)
        raw_name = splits[0]
        if len(splits) == 2:
            suffix = splits[1]
        return raw_name, suffix

    def _get_raw_name_from_input(self, name):
        suffix = "0"
        raw_name = name
        if re.match(r"^edge_[0-9]+_*", name):
            prefix = re.findall("edge_[0-9]+_", name)[0]
            raw_name = raw_name[len(prefix):]

        splits = raw_name.split(':', 1)
        raw_name = splits[0]
        if len(splits) == 2:
            suffix = splits[1]

        return raw_name, suffix

    def _get_partition_name_node_dict(self):
        partition_name_node_dict = dict()
        for graph in self.partition_graphs:
            for node in graph.node:
                name = node.name
                if name in partition_name_node_dict:
                    raise ValueError("name duplicate in partition graphs!" % name)
                partition_name_node_dict[name] = node
        return partition_name_node_dict

    def _get_graph_name_node_dict(self):
        graph_name_node_dict = dict()
        for node in self.graphdef.node:
            name = node.name
            graph_name_node_dict[name] = node
        return graph_name_node_dict


    def _is_same_device(self, device1, device2):
        if type(device1) in [unicode, str] and type(device2) in [unicode, str]:
            return device1 == device2
        elif type(device1) in [unicode, str]:
            device1 = self._get_structural_device(device1)
        elif type(device2) in [unicode, str]:
            device2 = self._get_structural_device(device2)
        return cmp(device1, device2) == 0

    def _is_same_worker(self, device1, device2):
        if type(device1) in [unicode, str]:
            device1 = self._get_structural_device(device1)
        if type(device2) in [unicode, str]:
            device2 = self._get_structural_device(device2)

        return (device1['job'] == device2['job']) and \
               (device1['replica'] == device2['replica']) and \
               (device1['task'] == device2['task'])

    def _is_GPU(self, device):
        if type(device) in [unicode, str]:
            device = self._get_structural_device(device)

        return device['device'] == 'GPU'

    def _get_structural_device(self, device):
        splits = device.split('/')
        for s in device.split('/'):
            if 'job' in s:
                job = s.split(':')[1]
            elif 'replica' in s:
                replica = s.split(':')[1]
            elif 'task' in s:
                task = s.split(':')[1]
            elif 'device' in s:
                ss = s.split(':')
                device_type = ss[1]
                device_id = ss[2]
        structural_device = {
            'job': job,
            'replica': replica,
            'task': task,
            'device': device_type,
            'device_id': device_id,
        }
        return structural_device


    def _get_send_node_dict(self):
        send_node_dict = dict()
        for name, node in self.partition_name_node_dict.items():
            if node.attr["tensor_name"].s and "Send" in node.op:
                send_node_dict[name] = node


        return send_node_dict


    def get_flops_from_profiler(self):
        name_flops = dict()
        ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder()
        ProfileOptionBuilder._options = ProfileOptionBuilder.float_operation()
        ProfileOptionBuilder.with_empty_output()
        GraphNodeProto = tf.profiler.profile(
            graph=self.graph,
            run_meta=self.run_metadata,
            options=ProfileOptionBuilder.build())
        for children in GraphNodeProto.children:
            name_flops[children.name] = children.float_ops
        return name_flops

    def export_partition_graphs(self, export_path):
        trace_file = open(export_path, 'w')
        trace_file.write(str(self.partition_graphs))
        trace_file.close()

    def export_cost_graph(self, export_path):
        cost_graph = self.cost_graph
        trace_file = open(export_path, 'w')
        trace_file.write(str(cost_graph))
        trace_file.close()

    def export_labelled_graph(self, export_path):
        labelled_graph = self.get_labelled_graph()
        trace_file = open(export_path, 'w')
        trace_file.write(str(labelled_graph))
        trace_file.close()

    # def get_first_n_op(self,op_dict,n):
    def _store_execution_time(self, raw_name, start_time, end_time):
        #if (self.name_cost_dict.has_key(raw_name)):
        #    start = min(start_time, self.name_cost_dict[raw_name][0])
        #    end = max(end_time, self.name_cost_dict[raw_name][1])
        #    self.name_cost_dict[raw_name] = (start, end)
        #else:
        #    self.name_cost_dict[raw_name] = (start_time, end_time)
        if (raw_name in self.name_cost_dict):
            self.name_cost_dict[raw_name]+=(end_time-start_time)
        else:
            self.name_cost_dict[raw_name] = end_time-start_time

    def calculate_op_cost(self):
        graph_name_node_dict = self.graph_name_node_dict
        self.name_cost_dict = dict()
        self.name_device_dict = dict()
        stat_name = dict()
        counter = 0
        counter1 = 0
        counter2 =0
        for dev_step_stat in self.step_stats.dev_stats:
            for step_stat in dev_step_stat.node_stats:
                raw_name, suffix = self._get_raw_name_from_step_stats(step_stat.node_name)
                stat_name[raw_name] = 1
        for name in graph_name_node_dict:
            if name in stat_name:
                counter = counter + 1
        print("Total nodes in graph:%d" % len(graph_name_node_dict))
        print("nodes in graph which estimated by step_stats:%d" % counter)
        print("Total nodes in step_stats:%d" % len(stat_name))
        print("Total nodes in partial graph:%d" % len(self.partition_name_node_dict.keys()))
        for name in self.partition_name_node_dict:
            if name not in graph_name_node_dict:
                counter1 += 1
        print("nodes in partition graph but not in graph:%d" % counter1)

        self.name_cost_dict = { x: [float('inf'), 0] for x in graph_name_node_dict.keys() }
        for dev in self.step_stats.dev_stats:
            for node in dev.node_stats:
                name = node.node_name.split(':')[0]
                if name in self.name_cost_dict.keys():
                    self.name_cost_dict[name][0] = min(self.name_cost_dict[name][0], node.all_start_micros)
                    self.name_cost_dict[name][1] = max(self.name_cost_dict[name][1], node.all_start_micros + node.all_end_rel_micros)

        print("Total nodes in cost_model:%d" % len(self.name_cost_dict))


        for name in graph_name_node_dict.keys():
            if name not in self.name_cost_dict:
                self.name_cost_dict[name] = [0,0]
        ret = dict()
        for name, cost in self.name_cost_dict.items():
        #    device = 'unknown'
        #    if self.partition_name_node_dict.has_key(name):
        #        partition_node = self.partition_name_node_dict[name]
        #        device = '/'+ partition_node.device.split('/')[-1]
            ret[name] = cost[1]-cost[0] if cost[1]-cost[0]>0 else 0
        return ret

    def _get_recv_tensor_dict(self):
        #key:tensor name
        #value:step_stat
        recv_tensor_dict = dict()
        for dev_step_stat in self.step_stats.dev_stats:
            for step_stat in dev_step_stat.node_stats:
                if step_stat.node_name!="RecvTensor":
                    continue
                key = step_stat.timeline_label.split(" ")[2]
                recv_tensor_dict[key] = step_stat
        return recv_tensor_dict


    def analysis_memcpy(self):

        device_speed = dict()
        device_size = dict()
        device_time = dict()
        send_node_dict = self._get_send_node_dict()
        print("11111111111111111")
        recv_tensor_dict = self._get_recv_tensor_dict()
        print("22222222222222222")
        for dev_step_stat in self.step_stats.dev_stats:
            for step_stat in dev_step_stat.node_stats:
                if "Send" not in step_stat.node_name:
                    continue
                raw_name, suffix = self._get_raw_name_from_step_stats(step_stat.node_name)

                send_node = send_node_dict.get(raw_name, None)
                if send_node == None:
                    print(raw_name)
                    print("recv_node == None")
                    continue
                recv_device = '/' + send_node.attr['recv_device'].s.decode().split('/')[-1]
                send_device = '/' + send_node.attr['send_device'].s.decode().split('/')[-1]
                recv_device = send_node.attr['recv_device'].s.decode()
                send_device = send_node.attr['send_device'].s.decode()
                key = send_device + recv_device
                recv_tensor = recv_tensor_dict.get(send_node.attr['tensor_name'].s,None)
                if not recv_tensor:
                    continue
                size = float(recv_tensor.output[0].tensor_description.allocation_description.requested_bytes)

                start_time = step_stat.all_start_micros
                end_time = recv_tensor.all_start_micros+recv_tensor.op_end_rel_micros
                time = end_time-start_time
                if time<10:
                    continue
                if device_size.get(key, None) == None:
                    device_size[key] = list()
                device_size[key].append(size)
                if device_time.get(key, None) == None:
                    device_time[key] = list()
                device_time[key].append(time)
        print("333333333333333333333")
        for key, size in device_size.items():
            print("start for")
            device_size[key].sort()
            device_time[key].sort()
            x = device_size[key]#[len(device_size[key])//2:]
            y = device_time[key]#[len(device_time[key])//2:]
            _min_val = min(device_time[key])
            #regr = linear_model.LinearRegression()
            print("start regression")
           # print(np.array(x).reshape(-1, 1))
            print(y)
           # regr.fit(np.array(x).reshape(-1, 1), y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(x), np.array(y))
            print("end regression")


            #device_speed[key] = (regr.coef_, regr.intercept_,_min_val)
            device_speed[key] = (slope, intercept, _min_val)

        print("4444444444444444444")
        self.device_speed = device_speed

        return device_speed



