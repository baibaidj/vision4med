

def multi_process_wrapper(num_thread = 10, verbose = True):
    """
    execute a function in multi processes, this wrapper makes several assumptions. 
    1. the first argument to the function is an iterables which will be splited into different processes as a guide. 
    2. the return objects of the function, if it is a dict, then its values must be list or tuple. 
    3. the function must be a stand-alone object instead of a class method that has to be called with a self handle. 

    """
    cpu_count = multiprocessing.cpu_count()
    if num_thread <= 0 or num_thread > cpu_count:
        num_thread = cpu_count

    is_dict = lambda x : isinstance(x, dict)
    def update_dict(holder, inputs):
        assert isinstance(holder, dict)
        assert isinstance(inputs, dict)
        
        for k, v in inputs.items():
            holder.setdefault(k, []).extend(v)
        # return holder

    def force2list(x):
        if isinstance(x, list):
            return x
        else:
            return list(x)
    
    def real_decorator(func_obj):
        def inter_logic(iterables, *args, **kwargs):
            sample_idx = sorted(list(iterables)) if is_dict(iterables) else list(range(len(iterables)))
            sample_num = len(sample_idx)
                        # func_name = 'new_func_%d'%i
            # class eval(func_name)(func_obj):
            #     def __init__(self, a = None):
            #         self.a = a
            #         super.__init__()
            # func = eval(func_name)()
            func = func_obj()
            if num_thread > 1:  #
                per_part = len(sample_idx) // num_thread + 1
                pool = multiprocessing.Pool(processes=num_thread)
                process_list = []
                for i in range(num_thread):
                    # if i in [18]:
                    start = int(i * per_part)
                    stop = int((i + 1) * per_part)
                    stop = min(stop, sample_num)
                    if verbose: print('thread=%d, start=%d, stop=%d' % (i, start, stop))
                    if not is_dict(iterables):
                        inter_iterables = [iterables[k] for k in sample_idx[start:stop]]
                    else:
                        inter_iterables = {k : iterables[k] for k in sample_idx[start:stop]}
                    this_proc = pool.apply_async(func, args=(inter_iterables, ) + args, kwds=kwargs)
                    process_list.append(this_proc)
                    # print('here')
                pool.close()
                pool.join()
                
                final_return = []
                for pix, proc in enumerate(process_list):
                    return_objs = proc.get()
                    if isinstance(return_objs, type(None)): return
                    for rix, reo in enumerate(return_objs):
                        if pix == 0:
                            if is_dict(reo):
                                final_return.append(dict()) # assume that the keys are list
                            else:
                                final_return.append(list())
                        if is_dict(reo):
                            update_dict(final_return[rix], reo)
                        else:
                            final_return[i].extend(force2list(reo))
            else:
                final_return = func(iterables, *args, **kwargs)
            return final_return
        return inter_logic
    return real_decorator


@multi_process_wrapper(num_thread=1)
class test_func():
    def __init__(self,  verbose = False):
        self.verbose = verbose

    def __call__(self, inputs = (1,2,3)):
        outputs = []
        for i in inputs:
            j = i+10
            print(i, j)
            outputs.append(j)
        return outputs
