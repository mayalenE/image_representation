import numpy as np

def get_sub_dictionary_variable(base_dict, variable):

    var_sub_elems = variable.split('.')

    cur_elem = base_dict
    for sub_elem in var_sub_elems:

        sub_elem_string_elements = sub_elem.split('[')
        sub_elem_basename = sub_elem_string_elements[0]

        if sub_elem_basename not in cur_elem:
            raise KeyError('Subelement {!r} does not exist in the base dictionary.'.format(sub_elem_basename))

        if len(sub_elem_string_elements) == 1:
            cur_elem = cur_elem[sub_elem_basename]
        else:
            sub_elem_index = sub_elem_string_elements[1].replace(']', '')

            try:
                sub_elem_index = int(sub_elem_index)
            except Exception as err:
                raise NotImplementedError('The sub indexing does only allow single indexes!') from err

            cur_elem = cur_elem[sub_elem_basename][sub_elem_index]

    return cur_elem

def do_filter_boolean(data, filter):

    if isinstance(filter, tuple):

        if len(filter) == 3:

            bool_component_1 = do_filter_boolean(data, filter[0])
            bool_component_2 = do_filter_boolean(data, filter[2])

            if filter[1] == 'and':
                ret_val = bool_component_1 & bool_component_2
            elif filter[1] == 'or':
                ret_val = bool_component_1 | bool_component_2
            elif filter[1] == '<':
                ret_val = bool_component_1 < bool_component_2
            elif filter[1] == '<=':
                ret_val = bool_component_1 <= bool_component_2
            elif filter[1] == '>':
                ret_val = bool_component_1 > bool_component_2
            elif filter[1] == '>=':
                ret_val = bool_component_1 >= bool_component_2
            elif filter[1] == '==':
                ret_val = bool_component_1 == bool_component_2
            elif filter[1] == '!=':
                ret_val = bool_component_1 != bool_component_2
            elif filter[1] == '+':
                ret_val = bool_component_1 + bool_component_2
            elif filter[1] == '-':
                ret_val = bool_component_1 - bool_component_2
            elif filter[1] == '*':
                ret_val = bool_component_1 * bool_component_2
            elif filter[1] == '/':
                ret_val = bool_component_1 / bool_component_2
            elif filter[1] == '%':
                ret_val = bool_component_1 % bool_component_2
            else:
                raise ValueError('Unknown operator {!r}!'.format(filter[1]))

        elif len(filter) == 2:

            val_component_1 = do_filter_boolean(data, filter[1])

            if filter[0] == 'sum':
                ret_val = np.sum(val_component_1)
            elif filter[0] == 'cumsum':
                ret_val = np.cumsum(val_component_1)
            elif filter[0] == 'max':
                ret_val = np.max(val_component_1)
            elif filter[0] == 'min':
                ret_val = np.min(val_component_1)
            else:
                raise ValueError('Unknown operator {!r}!'.format(filter[0]))

        else:
            raise ValueError('Unknown filter command {!r}!'.format(filter))

    else:

        is_var = False
        if isinstance(filter, str):

            # check if string is a variable in the data
            is_var = True
            try:

                if isinstance(data, list) or isinstance(data, np.ndarray):
                    get_sub_dictionary_variable(data[0], filter)
                else:
                    # check first item if the data object has a __iter__ method such as the explorationdatahandler
                    for item in data:
                        get_sub_dictionary_variable(item, filter)
                        break

            except KeyError:
                is_var = False

        if is_var:
            # if the string is a variable then get the data of the variable:
            ret_val = np.zeros(len(data))

            for data_idx, cur_data in enumerate(data):
                ret_val[data_idx] = get_sub_dictionary_variable(cur_data, filter)
        else:
            ret_val = filter

    return ret_val
