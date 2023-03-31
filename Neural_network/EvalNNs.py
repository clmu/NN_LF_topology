

class EvalNeuralNetworkEpochs:

    def __init__(self, list, name_list, remark_list):
        '''
        Initializes eval class.
        :param list: list of Neural networks to be evaluated.
                    All submitted networks need to have pre trained checkpoints.
        :param name_list:
        :param remark_list:
        '''

        if len(list) != len(name_list) or \
                len(list) != len(remark_list) or \
                    len(name_list) != len(remark_list):
            raise IndexError('Provided list are not of same length')
        self.nn_list = list
        self.nn_name_list = name_list
        self.nn_remark_list = remark_list
