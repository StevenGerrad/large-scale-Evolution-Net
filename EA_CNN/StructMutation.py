import random


class StructMutation():
    '''
    can mutate: hidden size, add edge, learning rate, add vertex, 
    '''
    def __init__(self):
        self._edge_types = []

    def mutate(self, dna):
        '''
        TODO: 可能出现由于概率'没有任何变异'的情况，不能让其发生
        1. 添加边时：添加identity, 则矩阵拼接时需要维度匹配 / 添加conv则需要是设置好参数
        '''
        mutated_dna = dna
        mutated_cnt = 0
        while mutated_cnt == 0:
            # 1. Try the candidates in random order until one has the right connectivity.(Add)
            for from_vertex_id, to_vertex_id in self._vertex_pair_candidates(dna):
                # 防止每次变异次数过多
                if random.random() < pow(0.4, mutated_cnt + 1):
                    mutated_cnt += 1
                    self._mutate_structure(mutated_dna, from_vertex_id, to_vertex_id)

            # 2. Try to mutate learning Rate
            self.mutate_learningRate(mutated_dna)

            # 3. mutate the hidden layer's size
            # self.mutate_hidden_size(dna)

            # 4. Mutate the vertex (Add)
            # self.mutate_vertex(mutated_dna)
            if random.random() > 0.6:
                mutated_cnt += 1
                self.mutate_vertex(mutated_dna)
        return mutated_dna

    def _vertex_pair_candidates(self, dna):
        """Yields connectable vertex pairs."""
        from_vertex_ids = self._find_allowed_vertices(dna)
        # if not from_vertex_ids: raise exceptions.MutationException(), 打乱次序
        random.shuffle(from_vertex_ids)

        to_vertex_ids = self._find_allowed_vertices(dna)
        # if not to_vertex_ids: raise exceptions.MutationException()
        random.shuffle(to_vertex_ids)

        for to_vertex_id in to_vertex_ids:
            # Avoid back-connections. TODO: 此处可能会涉及到 拓扑图的顺序判断
            # disallowed_from_vertex_ids, _ = topology.propagated_set(to_vertex_id)
            disallowed_from_vertex_ids = self._find_disallowed_from_vertices(dna, to_vertex_id)
            for from_vertex_id in from_vertex_ids:
                if from_vertex_id in disallowed_from_vertex_ids:
                    continue
                # This pair does not generate a cycle, so we yield it.
                yield from_vertex_id, to_vertex_id

    def _find_allowed_vertices(self, dna):
        ''' TODO: 除第一层(假节点)外的所有vertex_id '''
        return list(range(0, len(dna.vertices)))

    def _find_disallowed_from_vertices(self, dna, to_vertex_id):
        ''' 寻找不可作为起始层索引的：反向链接的，重复连接的Edge '''
        res = list(range(to_vertex_id, len(dna.vertices)))
        # 排查每个 vertex 是否不符合, 即索引在前面的 vertex 的所有 edges_out
        for i, vertex in enumerate(dna.vertices[:to_vertex_id]):
            for edge in vertex.edges_out:
                if dna.vertices.index(edge.to_vertex) == to_vertex_id:
                    if i not in res:
                        res.append(i)
                        continue
        return res

    def _mutate_structure(self, dna, from_vertex_id, to_vertex_id):
        """Adds the edge to the DNA instance."""
        if dna.has_edge(from_vertex_id, to_vertex_id):
            return False
        else:
            # TODO: edge 有两个类型，identity 和 conv (主要调节 stride, 在默认padding补全的情况下)
            # 1. 若数据维度不变，可以用identity， 则需要检查 stride 是否不变
            res = True
            bin_stride = 0
            for vertex_id, vert in enumerate(dna.vertices[from_vertex_id + 1:to_vertex_id],
                                             start=from_vertex_id + 1):
                edg_direct = None
                for edg in vert.edges_in:
                    if edg.from_vertex == dna.vertices[
                            vertex_id - 1] and edg.to_vertex == dna.vertices[vertex_id]:
                        edg_direct = edg
                        break
                if edg_direct.stride_scale != 0:
                    res = False
                    bin_stride += edg_direct.stride_scale
            if res and random.random() > 0.6:
                print("[add_edge]->identity:", from_vertex_id, to_vertex_id)
                new_edge = dna.add_edge(from_vertex_id, to_vertex_id)
                return res
            # 2. 若数据维度改变(变小)，要用conv
            print("[add_edge]->conv:", from_vertex_id, to_vertex_id)
            depth_f = max(1.0, random.random() * 4)
            filter_h = 1
            filter_w = 1
            new_edge = dna.add_edge(from_vertex_id,
                                    to_vertex_id,
                                    edge_type='conv',
                                    depth_factor=depth_f,
                                    filter_half_height=filter_h,
                                    filter_half_width=filter_w,
                                    stride_scale=bin_stride)
            return True

    def mutate_hidden_size(self, dna):
        '''
        TODO: mutate the hidden layer's size 
        高斯分布随机生成, 对所有 hidden layer 变动...不可取
        '''
        # for i in list(range(1, len(dna.vertices) - 1)):
        #     if random.random() > 0.6:
        #         last = dna.vertices[i].outputs_mutable
        #         before = dna.vertices[i - 1].outputs_mutable
        #         after = dna.vertices[i + 1].outputs_mutable

        #         alpha = min(before - last, last - after) / 3
        #         next = last + alpha * np.random.randn(1)
        #         next = int(next[0])
        #         if next > before:
        #             next = before
        #         elif next < after:
        #             next = after
        #         dna.vertices[i].outputs_mutable = next

    def mutate_learningRate(self, dna):
        # mutated_dna = copy.deepcopy(dna)
        mutated_dna = dna
        # Mutate the learning rate by a random factor between 0.5 and 2.0,
        # uniformly distributed in log scale.
        factor = 2**random.uniform(-1.0, 1.0)
        mutated_dna.learning_rate = dna.learning_rate * factor
        return mutated_dna

    def mutate_vertex(self, dna):
        # mutated_dna = copy.deepcopy(dna)
        mutated_dna = dna
        # 随机选择一个 vertex_id 插入 vertex
        after_vertex_id = random.choice(self._find_allowed_vertices(dna))
        if after_vertex_id == 0:
            return mutated_dna

        print('outputs_mutable', dna.vertices[after_vertex_id].outputs_mutable,
              dna.vertices[after_vertex_id - 1].outputs_mutable)
        # TODO: how it supposed to mutate
        vertex_type = 'linear'
        if random.random() > 0.2:
            vertex_type = 'bn_relu'

        edge_type = 'identity'
        if random.random() > 0.2:
            edge_type = 'conv'

        mutated_dna.add_vertex(after_vertex_id, vertex_type, edge_type)
        return mutated_dna
